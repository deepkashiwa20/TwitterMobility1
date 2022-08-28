# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:28:06 2020
@author: wb
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torchsummary import summary


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        K: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        V: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        attn_mask: [batch_size, n_heads, seq_len, seq_len] 可能没有
        '''
        B, n_heads, len1, len2, d_k = Q.shape 
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) 
        # scores : [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), N(Spatial) or T(Temporal)]
        # scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]]
        return context


class TMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(TMultiHeadAttention, self).__init__()
        
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
            

        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, N, T, C]
        input_K: [batch_size, N, T, C]
        input_V: [batch_size, N, T, C]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        B, N, T, C = input_Q.shape
        # [B, N, T, C] --> [B, N, T, h * d_k] --> [B, N, T, h, d_k] --> [B, h, N, T, d_k]
        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4) # Q: [B, h, N, T, d_k]
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # K: [B, h, N, T, d_k]
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # V: [B, h, N, T, d_k]

        # attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context = ScaledDotProductAttention()(Q, K, V) #[B, h, N, T, d_k]
        context = context.permute(0, 2, 3, 1, 4) #[B, N, T, h, d_k]
        context = context.reshape(B, N, T, self.heads * self.head_dim) # [B, N, T, C]
        # context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc_out(context) # [batch_size, len_q, d_model]
        return output 

class Transformer_EncoderBlock(nn.Module):
    def __init__(self, embed_size, pe_length, heads ,forward_expansion, gpu, dropout):
        super(Transformer_EncoderBlock, self).__init__()
        
        # Temporal embedding One hot
        self.pe_length = pe_length
#         self.one_hot = One_hot_encoder(embed_size, pe_length)          # temporal embedding by one-hot
        self.temporal_embedding = nn.Embedding(pe_length, embed_size)  # temporal embedding  by nn.Embedding
        self.attention = TMultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)
        self.gpu = gpu
    def forward(self, value, key, query):
        B, N, T, C = query.shape
        
#         D_T = self.one_hot(t, N, T)                          # temporal embedding by one-hot
        D_T = self.temporal_embedding(torch.arange(0, T).to(self.gpu))    # temporal embedding  by nn.Embedding
        D_T = D_T.expand(B, N, T, C)

        # temporal embedding + query。 (concatenated or add here is add)
        query = query + D_T  
#         print('query + D_T shape:',query.shape)

        attention = self.attention(query, query, query)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

### TTransformer_EncoderLayer
class TTransformer_EncoderLayer(nn.Module):
    def __init__(self, embed_size, pe_length, heads ,forward_expansion, gpu, dropout):
        super(TTransformer_EncoderLayer, self).__init__()
#         self.STransformer = STransformer(embed_size, heads, adj, cheb_K, dropout, forward_expansion)
        self.Transformer_EncoderBlock = Transformer_EncoderBlock(embed_size, pe_length, heads ,forward_expansion, gpu, dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, value, key, query):
    # value,  key, query: [N, T, C] [B, N, T, C]
        # Add skip connection,run through normalization and finally dropout
        x1 = self.norm1(self.Transformer_EncoderBlock(value, key, query) + query) #(B, N, T, C)
        x2 = self.dropout(x1) 

        return x2

### Encoder
class Encoder(nn.Module):
    # 堆叠多层 ST-Transformer Block
    def __init__(
        self,embed_size,num_layers,pe_length,heads,forward_expansion,gpu,dropout):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.gpu = gpu
        self.layers = nn.ModuleList([ TTransformer_EncoderLayer(embed_size, pe_length, heads ,forward_expansion, gpu, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):     
        # x: [N, T, C]  [B, N, T, C]
        out = self.dropout(x)      
        # In the Encoder the query, key, value are all the same.
        for layer in self.layers:
            out = layer(out, out, out)
        return out     
    
### Transformer   
class T_Transformer_block(nn.Module):
    def __init__(self,embed_size,num_layers,pe_length,heads,forward_expansion, gpu,dropout):
        super(T_Transformer_block, self).__init__()
        self.encoder = Encoder(embed_size,num_layers,pe_length,heads,forward_expansion,gpu,dropout)
        self.gpu = gpu

    def forward(self, src): 
        ## scr: [N, T, C]   [B, N, T, C]
        enc_src = self.encoder(src) 
        return enc_src # [B, N, T, C]


class Transformer(nn.Module):
    def __init__(
        self, in_channels, out_channels, embed_size, pe_length, num_layers, timestep_in, timestep_out, heads, forward_expansion, gpu, dropout):        
        super(Transformer, self).__init__()

        self.forward_expansion = forward_expansion
        # C --> expand  --> hidden dim (embed_size)
        self.conv1 = nn.Conv2d(in_channels, embed_size, 1)
        
        self.T_Transformer_block = T_Transformer_block(embed_size, num_layers, pe_length, heads, forward_expansion, gpu, dropout)

        # Reduce the temporal dimension。  timestep_in --> out_timestep_in
        self.conv2 = nn.Conv2d(timestep_in, timestep_out, 1)  
        # Reduce the C dimension，to 1。
        self.conv3 = nn.Conv2d(embed_size, out_channels, 1)
        self.relu = nn.ReLU()
        self.Tanh = nn.Tanh()
    
    def forward(self, x):
        # input x shape  [B, T, N, C]  C  = CHANNEL = 1
        # C: channel。  N:nodes。  T:time
        
        x = x.permute(0,3,2,1)   # [B, T, N, C] -> [B, C, N, T]
        input_Transformer = self.conv1(x)     #    x shape[B, C, N, T]   --->    input_Transformer shape： [B, H = embed_size = 64, N, T] 
        input_Transformer = input_Transformer.permute(0, 2, 3, 1)    # [B, H, N, T] [B, N, T, H]
        output_Transformer = self.T_Transformer_block(input_Transformer)  # [B, N, T, H]
        output_Transformer = output_Transformer.permute(0, 2, 1, 3)   # [B, N, T, H] -> [B, T, N, H]
        
        out = self.relu(self.conv2(output_Transformer))    #   [B, T, N, H] ->  [B, T, N, C=1]         
        out = out.permute(0, 3, 2, 1)           # [B, T, N, C=1]  ->  [B, C=1, N, T]
        out = self.conv3(out)                   # 
        out = out.permute(0, 3, 2, 1)           # [B, C=1, N, T] -> [B,T,N,1]

        return out      #[B, TIMESTEP_OUT, N, C]   C = CHANNEL = 1


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


class gatedFusion(nn.Module):
    """
    gated fusion
    HS:     [batch_size, num_step, num_vertex, D]
    HT:     [batch_size, num_step, num_vertex, D]
    D:      output dims
    return: [batch_size, num_step, num_vertex, D]
    """

    def __init__(self, D, m, bn_decay):
        super(gatedFusion, self).__init__()
        self.FC_xs = FC(input_dims=D, units=D, activations=None,
                        bn_decay=bn_decay, use_bias=False)
        self.FC_xt = FC(input_dims=D, units=D, activations=None,
                        bn_decay=bn_decay, use_bias=True)
        self.FC_h = FC(input_dims=[D, D], units=[D, D+m], activations=[F.relu, None],
                       bn_decay=bn_decay)

    def forward(self, HS, HT):
        XS = self.FC_xs(HS)
        XT = self.FC_xt(HT)
        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.mul(z, HS), torch.mul(1 - z, HT))
        H = self.FC_h(H)
        del XS, XT, z
        return H


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
        self.FC_k = FC(input_dims=D+m, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_v = FC(input_dims=D+m, units=D, activations=F.relu,
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


class Transformer_CMem(nn.Module):
    def __init__(
            self, n_nodes, in_channels, out_channels, embed_size, pe_length, num_layers, timestep_in, timestep_out, heads,
            forward_expansion, gpu, dropout, mem_num:int=6, mem_dim:int=8):
        super(Transformer_CMem, self).__init__()

        self.num_his = timestep_in
        self.num_pred = timestep_out
        self.forward_expansion = forward_expansion
        # C --> expand  --> hidden dim (embed_size)
        self.conv1 = nn.Conv2d(in_channels, embed_size, 1)
        self.T_Transformer_block = T_Transformer_block(embed_size, num_layers, pe_length, heads, forward_expansion, gpu, dropout)

        #self.conv_tw = nn.Conv2d(in_channels, embed_size, 1)
        #self.T_Transformer_block_tw = T_Transformer_block(embed_size, num_layers, pe_length, heads, forward_expansion, gpu, dropout)

        # Reduce the temporal dimension。  timestep_in --> out_timestep_in
        self.transformAttention = transformAttention(8, 8, mem_dim, 0.1)
        self.conv2 = nn.Conv2d(timestep_in, timestep_out, 1)
        # Reduce the C dimension，to 1。
        self.conv3 = nn.Conv2d(embed_size, out_channels, 1)
        self.relu = nn.ReLU()
        self.Tanh = nn.Tanh()
        # memory use
        self.N = n_nodes
        self.D = embed_size
        self.mem_num = mem_num
        self.mem_dim = mem_dim
        self.st_memory = self.construct_st_memory()  # local memory: frame-wise
        self.seq_memory = self.construct_seq_memory()  # global memory: seq-wise
        # memory fusion
        self.mem_fus = gatedFusion(embed_size+mem_dim, 0, 0.1)

    def construct_seq_memory(self):
        memory_weight = nn.ParameterDict()
        memory_weight['memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)     # (M, d)
        nn.init.xavier_normal_(memory_weight['memory'])
        memory_weight['Wa'] = nn.Parameter(torch.randn(self.num_pred*self.N*self.D, self.mem_dim), requires_grad=True)    # for project to query
        nn.init.xavier_normal_(memory_weight['Wa'])
        memory_weight['fc'] = nn.Parameter(torch.randn(self.mem_dim, self.mem_dim*self.N*self.num_pred), requires_grad=True)
        nn.init.xavier_normal_(memory_weight['fc'])
        return memory_weight

    def query_seq_memory(self, h_t:torch.Tensor):
        assert len(h_t.shape) == 4, 'Input to query Seq-Memory must be a 4D tensor'

        h_t_flat = h_t.reshape(h_t.shape[0], -1)    # (B, T*N*h)
        query = torch.mm(h_t_flat, self.seq_memory['Wa'])     # (B, d)
        att_score = torch.softmax(torch.mm(query, self.seq_memory['memory'].t()), dim=1)         # alpha: (B, M)
        proto_t = torch.mm(att_score, self.seq_memory['memory'])    # (B, d)
        mem_t = torch.mm(proto_t, self.seq_memory['fc'])   # (B, T*N*d)
        _h_t = torch.cat([h_t, mem_t.reshape(h_t.shape[0], self.num_pred, self.N, self.mem_dim)], dim=-1)      # (B, T, N, h+d)
        return _h_t

    def construct_st_memory(self):
        memory_weight = nn.ParameterDict()
        memory_weight['memory'] = nn.Parameter(torch.randn(self.N, self.mem_num, self.mem_dim), requires_grad=True)     # (N, M, d)
        nn.init.xavier_normal_(memory_weight['memory'])
        memory_weight['Wa'] = nn.Parameter(torch.randn(self.D, self.mem_dim), requires_grad=True)    # for project to query
        nn.init.xavier_normal_(memory_weight['Wa'])
        memory_weight['fc'] = nn.Parameter(torch.randn(self.mem_dim, self.mem_dim), requires_grad=True)
        nn.init.xavier_normal_(memory_weight['fc'])
        return memory_weight

    def query_st_memory(self, h_t:torch.Tensor):
        assert len(h_t.shape) == 3, 'Input to query ST-Memory must be a 3D tensor'

        query = torch.einsum('bnh,hd->bnd', h_t, self.st_memory['Wa'])      # (B, N, d)
        att_score = torch.softmax(torch.einsum('bnd,nmd->bnm', query, self.st_memory['memory']), dim=-1)  # alpha: (B, N, M)
        proto_t = torch.einsum('bnm,nmd->bnd', att_score, self.st_memory['memory'])      # (B, N, d)
        mem_t = torch.matmul(proto_t, self.st_memory['fc'])     # (B, N, d)
        _h_t = torch.cat([h_t, mem_t], dim=-1)      # (B, N, h+d)
        return _h_t

    def forward(self, x):
        # input x shape  [B, T, N, C]  C  = CHANNEL = 1
        # C: channel。  N:nodes。  T:time
        x, tw = torch.unsqueeze(x[..., 0], -1), torch.unsqueeze(x[..., -1], -1)  # channel 0: mob 1: twit

        x = x.permute(0, 3, 2, 1)  # [B, T, N, C] -> [B, C, N, T]
        input_Transformer = self.conv1(x)  # x shape[B, C, N, T]   --->    input_Transformer shape： [B, H = embed_size = 64, N, T]
        input_Transformer = input_Transformer.permute(0, 2, 3, 1)  # [B, H, N, T] [B, N, T, H]
        output_Transformer = self.T_Transformer_block(input_Transformer)  # [B, N, T, H]
        output_Transformer = output_Transformer.permute(0, 2, 1, 3)  # [B, N, T, H] -> [B, T, N, H]

        # encode twitter
        # tw = tw.permute(0, 3, 2, 1)  # [B, T, N, C] -> [B, C, N, T]
        # input_tw = self.conv_tw(tw)  # x shape[B, C, N, T]   --->    input_Transformer shape： [B, H = embed_size = 64, N, T]
        # input_tw = input_tw.permute(0, 2, 3, 1)  # [B, H, N, T] [B, N, T, H]
        # output_tw = self.T_Transformer_block_tw(input_tw)  # [B, N, T, H]
        # output_tw = output_tw.permute(0, 2, 1, 3)  # [B, N, T, H] -> [B, T, N, H]

        # query seq memory
        # output_1 = self.query_seq_memory(output_tw)
        # query st memory
        # x = []
        # for t in range(self.num_pred):
        #     x.append(self.query_st_memory(output_Transformer[:,t,:,:]))
        # output_2 = torch.stack(x, dim=1)
        #output_Transformer = self.mem_fus(output_1, output_2)
        #output_Transformer = torch.mul(output_2, torch.sigmoid(output_1))

        # query seq memory
        out_1 = self.query_seq_memory(output_Transformer)
        # query st memory
        x = []
        for t in range(self.num_pred):
            x.append(self.query_st_memory(output_Transformer[:, t, :, :]))
        out_2 = torch.stack(x, dim=1)
        out = self.mem_fus(out_1, out_2)
        #out = torch.mul(out_2, torch.sigmoid(out_1))

        #out = self.relu(self.conv2(output_Transformer))  # [B, Tin, N, H] ->  [B, Tout, N, H]
        #out = out.permute(0, 3, 2, 1)  # [B, T, N, C=1]  ->  [B, C=1, N, T]
        #query_tw = self.query_seq_memory(output_tw)
        out = self.transformAttention(out, out, out)
        out = out.permute(0, 3, 2, 1)  # [B, T, N, C=1]  ->  [B, C=1, N, T]
        out = self.conv3(out)  #
        out = out.permute(0, 3, 2, 1)  # [B, C=1, N, T] -> [B,T,N,1]

        return out  # [B, TIMESTEP_OUT, N, C]   C = CHANNEL = 1


def print_params(model_name, model):
    param_count=0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count += param.numel()
    print(f'{model_name}, {param_count} trainable parameters in total.')
    return  

def main():
    channel, his_len, seq_len=1, 12, 12
    GPU = '0'
    device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
    model = Transformer(in_channels=channel, 
                        embed_size=64, 
                        pe_length=his_len, 
                        num_layers=1, 
                        timestep_in=his_len, 
                        timestep_out=seq_len, 
                        heads=8, 
                        forward_expansion=64, 
                        gpu=device, 
                        dropout=0).to(device)
    print_params('Transformer', model)
    summary(model, (his_len, 47, channel), device=device)
    
if __name__ == '__main__':
    main()
