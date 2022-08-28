import torch
import torch.nn.functional as F
import torch.nn as nn
import math


class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
    def forward(self, x, node_embeddings):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        #default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                       #N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out
        return x_gconv

class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
    
class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, torch.stack(output_hidden, dim=0)

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)

class AGCRN(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, horizon, rnn_units=64, num_layers=2, default_graph=True, embed_dim=8, cheb_k=2):
        super(AGCRN, self).__init__()
        self.num_node = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = rnn_units
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers

        self.default_graph = default_graph
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, embed_dim), requires_grad=True)

        self.encoder = AVWDCRNN(num_nodes, input_dim, rnn_units, cheb_k, embed_dim, num_layers)

        #predictor
        self.end_conv = nn.Conv2d(1, self.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    # def forward(self, source, targets, teacher_forcing_ratio=0.5):
    def forward(self, source):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)

        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings)      #B, T, N, hidden
        output = output[:, -1:, :, :]                                   #B, 1, N, hidden

        #CNN based predictor
        output = self.end_conv((output))                         #B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)                             #B, T, N, C

        return output


class conv2d_(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size, stride=(1, 1),
                 padding='SAME', use_bias=True, activation=F.relu,
                 bn_decay=None):
        super(conv2d_, self).__init__()
        self.activation = activation
        if padding == 'SAME':
            self.padding_size = math.ceil(kernel_size)
        else:
            self.padding_size = [0, 0]
        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride,
                              padding=0, bias=use_bias)
        self.batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay)
        torch.nn.init.xavier_uniform_(self.conv.weight)

        if use_bias:
            torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = F.pad(x, ([self.padding_size[1], self.padding_size[1], self.padding_size[0], self.padding_size[0]]))
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = F.relu_(x)
        return x.permute(0, 3, 2, 1)


class FC(nn.Module):
    def __init__(self, input_dims, units, activations, bn_decay, use_bias=True):
        super(FC, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert type(units) == list
        self.convs = nn.ModuleList([conv2d_(
            input_dims=input_dim, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation,
            bn_decay=bn_decay) for input_dim, num_unit, activation in
            zip(input_dims, units, activations)])

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


class transformAttention(nn.Module):
    """
    transform attention mechanism
    X:          [batch_size, num_his, num_vertex, D]
    STE_his:    [batch_size, num_his, num_vertex, D]
    STE_pred:   [batch_size, num_pred, num_vertex, D]
    K:          number of attention heads
    d:          dimension of each attention outputs
    return:     [batch_size, num_pred, num_vertex, D]
    """

    def __init__(self, K, d, m, bn_decay):
        super(transformAttention, self).__init__()
        D = K * d
        self.K = K
        self.d = d
        self.FC_q = FC(input_dims=D+m, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_k = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_v = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu,
                     bn_decay=bn_decay)

    def forward(self, X, STE_his, STE_pred):
        batch_size = X.shape[0]
        # [batch_size, num_step, num_vertex, K * d]
        query = self.FC_q(STE_pred)
        key = self.FC_k(STE_his)
        value = self.FC_v(X)
        # [K * batch_size, num_step, num_vertex, d]
        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)
        # query: [K * batch_size, num_vertex, num_pred, d]
        # key:   [K * batch_size, num_vertex, d, num_his]
        # value: [K * batch_size, num_vertex, num_his, d]
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        # [K * batch_size, num_vertex, num_pred, num_his]
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        # [batch_size, num_pred, num_vertex, D]
        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)
        X = self.FC(X)
        del query, key, value, attention
        return X


class STEmbedding(nn.Module):
    """
    spatio-temporal embedding
    SE:     [num_vertex, D]
    TE:     [batch_size, num_hist + num_pred, 2] (dayofweek, timeofday)
    T:      num of time steps in one day
    D:      output dims
    return: [batch_size, num_his + num_pred, num_vertex, D]
    """

    def __init__(self, SE_dim, TE_dim, D, bn_decay):
        super(STEmbedding, self).__init__()
        self.FC_se = FC(
            input_dims=[SE_dim, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay
        )

        self.FC_te = FC(
            input_dims=[TE_dim, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay
        )

    def forward(self, SE, TE):
        SE = SE.unsqueeze(0).unsqueeze(0)
        SE = self.FC_se(SE)
        TE = TE.unsqueeze(dim=2)
        TE = self.FC_te(TE)
        return SE + TE


class AGCRN_CMem(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, horizon, rnn_units=64, num_layers=2, default_graph=True, embed_dim=8, cheb_k=2,
                 mem_num:int=6, mem_dim:int=8):
        super(AGCRN_CMem, self).__init__()
        self.num_node = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = rnn_units
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers

        self.default_graph = default_graph
        self.embed_dim = embed_dim
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, embed_dim), requires_grad=True)
        self.STEmbedding = STEmbedding(8, 64, 64, 0.1)

        self.encoder = AVWDCRNN(num_nodes, input_dim, rnn_units, cheb_k, embed_dim, num_layers//2)
        self.encoder_tw = AVWDCRNN(num_nodes, input_dim, rnn_units, cheb_k, embed_dim, num_layers//2)
        self.transformAttention = transformAttention(8, 8, 0, 0.1)
        self.decoder = AVWDCRNN(num_nodes, rnn_units+mem_dim, rnn_units, cheb_k, embed_dim, num_layers//2)
        self.FC = FC(input_dims=[rnn_units, rnn_units], units=[rnn_units, output_dim], activations=[F.relu, None], bn_decay=0.1)
        # memory use
        self.mem_num = mem_num
        self.mem_dim = mem_dim
        self.st_memory = self.construct_st_memory()  # local clustered memory: frame-wise
        self.seq_memory = self.construct_seq_memory()  # global memory: seq-wise

    def construct_seq_memory(self):
        memory_weight = nn.ParameterDict()
        memory_weight['memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)     # (M, d)
        nn.init.xavier_normal_(memory_weight['memory'])
        memory_weight['Wa'] = nn.Parameter(torch.randn(self.horizon*self.num_node*self.hidden_dim, self.mem_dim), requires_grad=True)    # for project to query
        nn.init.xavier_normal_(memory_weight['Wa'])
        memory_weight['fc'] = nn.Parameter(torch.randn(self.mem_dim, self.mem_dim*self.num_node*self.horizon), requires_grad=True)
        nn.init.xavier_normal_(memory_weight['fc'])
        return memory_weight

    def query_seq_memory(self, h_t:torch.Tensor):
        assert len(h_t.shape) == 4, 'Input to query Seq-Memory must be a 4D tensor'

        h_t_flat = h_t.reshape(h_t.shape[0], -1)    # (B, T*N*h)
        query = torch.mm(h_t_flat, self.seq_memory['Wa'])     # (B, d)
        att_score = torch.softmax(torch.mm(query, self.seq_memory['memory'].t()), dim=1)         # alpha: (B, M)
        proto_t = torch.mm(att_score, self.seq_memory['memory'])    # (B, d)
        mem_t = torch.mm(proto_t, self.seq_memory['fc'])   # (B, T*N*d)
        _h_t = torch.cat([h_t, mem_t.reshape(h_t.shape[0], self.horizon, self.num_node, self.mem_dim)], dim=-1)      # (B, T, N, h+d)
        return _h_t

    def construct_st_memory(self):      # clustered memory
        memory_weight = nn.ParameterDict()
        memory_weight['mem_pool'] = nn.Parameter(torch.randn(self.embed_dim, self.mem_num, self.mem_dim), requires_grad=True)     # (N, M, d)
        nn.init.xavier_normal_(memory_weight['mem_pool'])
        memory_weight['Wa'] = nn.Parameter(torch.randn(self.hidden_dim, self.mem_dim), requires_grad=True)    # for project to query
        nn.init.xavier_normal_(memory_weight['Wa'])
        memory_weight['fc'] = nn.Parameter(torch.randn(self.mem_dim, self.mem_dim), requires_grad=True)
        nn.init.xavier_normal_(memory_weight['fc'])
        return memory_weight

    def query_st_memory(self, h_t:torch.Tensor):
        assert len(h_t.shape) == 3, 'Input to query ST-Memory must be a 3D tensor'

        recon_mem = torch.einsum('ne,emd->nmd', self.node_embeddings, self.st_memory['mem_pool'])  # (N, M, d)
        query = torch.einsum('bnh,hd->bnd', h_t, self.st_memory['Wa'])      # (B, N, d)
        att_score = torch.softmax(torch.einsum('bnd,nmd->bnm', query, recon_mem), dim=-1)  # alpha: (B, N, M)
        proto_t = torch.einsum('bnm,nmd->bnd', att_score, recon_mem)      # (B, N, d)
        mem_t = torch.matmul(proto_t, self.st_memory['fc'])     # (B, N, d)
        _h_t = torch.cat([h_t, mem_t], dim=-1)      # (B, N, h+d)
        return _h_t

    # def forward(self, source, targets, teacher_forcing_ratio=0.5):
    def forward(self, source):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)

        X, TW = torch.unsqueeze(source[..., 0], -1), torch.unsqueeze(source[..., -1], -1)  # channel 0: mob 1: twit

        init_state = self.encoder.init_hidden(X.shape[0])
        X, states = self.encoder(X, init_state, self.node_embeddings)      #B, Tin, N, hidden
        TW, _ = self.encoder(TW, init_state, self.node_embeddings)       # B, Tin, N, hidden
        #STE = self.STEmbedding(self.node_embeddings, TW)

        #STEmem = self.query_seq_memory(STE)
        X = self.transformAttention(X, X, X)     #B, Tout, N, hidden

        # query seq memory
        X_seq = self.query_seq_memory(X)
        # query ST memory
        x = []
        for t in range(self.horizon):
            x.append(self.query_st_memory(X[:,t,:,:]))
        X_st = torch.stack(x, dim=1)
        X = torch.mul(X_st, torch.sigmoid(X_seq))

        output, _ = self.decoder(X, states, self.node_embeddings)   #B, Tout, N, hidden
        output = self.FC(output)
        return output


def main():
    import sys
    from torchsummary import summary
    from Param import CHANNEL,N_NODE,TIMESTEP_IN,TIMESTEP_OUT
    GPU = sys.argv[-1] if len(sys.argv) == 2 else '3'
    device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
    model = AGCRN(num_nodes=N_NODE, input_dim=CHANNEL, output_dim=CHANNEL, horizon=TIMESTEP_OUT).to(device)
    summary(model, (TIMESTEP_IN, N_NODE, CHANNEL), device=device)
    
if __name__ == '__main__':
    main()
