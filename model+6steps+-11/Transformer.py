import sys
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import Variable
from typing import List, Tuple
import numpy as np
from collections import defaultdict
from torchsummary import summary

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class Attention(nn.Module):
    def __init__(self, num_nodes, in_channels, key_kernel_size, stnorm_bool, snorm_bool, tnorm_bool, device):
        super(Attention, self).__init__()
        hidden_channels = in_channels // 2
        key_padding = key_kernel_size - 1
        self.snorm_bool = snorm_bool
        self.tnorm_bool = tnorm_bool
        self.stnorm_bool = stnorm_bool
        if tnorm_bool:
            self.tn = nn.BatchNorm1d(num_nodes * in_channels, track_running_stats=False, affine=True)
        if snorm_bool:
            self.sn = nn.InstanceNorm1d(in_channels, track_running_stats=False, affine=True)
        if stnorm_bool:
            self.stn = nn.InstanceNorm2d(in_channels, track_running_stats=False, affine=True)
        num = int(tnorm_bool) + int(snorm_bool) + int(stnorm_bool) + 1

        self.q_W = nn.Conv1d(num * in_channels, hidden_channels, key_kernel_size, 1, key_padding, bias=False)       # enhancing locality: NIPS'19
        self.k_W = nn.Conv1d(num * in_channels, hidden_channels, key_kernel_size, 1, key_padding, bias=False)
        self.v_W = nn.Conv1d(num * in_channels, hidden_channels, 1, bias=False)
        self.o_W = nn.Conv1d(hidden_channels, in_channels, 1, bias=False)
        self.ff_W = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 1),
            nn.ReLU(),
            nn.Conv1d(in_channels, in_channels, 1)
        )
        self.chomp = Chomp1d(key_padding)
        self.device = device

    def forward(self, input):
        b, c, n, t = input.shape

        x = input
        x_list = [x]
        if self.tnorm_bool:
            x_tnorm = self.tn(x.reshape(b, c * n, t)).view(b, c, n, t)
            x_list.append(x_tnorm)
        if self.snorm_bool:
            x_snorm = self.sn(x.permute(0, 3, 1, 2).reshape(b * t, c, n)).view(b, t, c, n).permute(0, 2, 3, 1)
            x_list.append(x_snorm)
        if self.stnorm_bool:
            x_stnorm = self.stn(x)
            x_list.append(x_stnorm)
        x = torch.cat(x_list, dim=1)
        x_f = x.permute(0, 2, 1, 3).reshape(b * n, -1, t)
        input_f = input.permute(0, 2, 1, 3).reshape(b * n, -1, t)

        q = self.chomp(self.q_W(x_f))
        k = self.chomp(self.k_W(x_f))
        v = self.v_W(x_f)
        attn = torch.bmm(q.permute(0, 2, 1), k)
        upper_mask = torch.triu(torch.ones(b * n, t, t), diagonal=1).to(self.device)
        attn = attn - 1000 * upper_mask         # restrict the attention to be unidirectional?
        attn = torch.softmax(attn, dim=-1)
        attn_out = torch.bmm(attn, v.permute(0, 2, 1)).permute(0, 2, 1)
        out_f = input_f + self.o_W(attn_out)
        out_f = out_f + self.ff_W(out_f)
        out = out_f.view(b, n, -1, t).permute(0, 2, 1, 3).contiguous()
        return out



class Transformer(nn.Module):
    def __init__(self, num_nodes, in_channels, out_channels, n_his, n_pred, hidden_channels, n_layers, device, stnorm_bool=False, snorm_bool=False, tnorm_bool=False, ext=False, daily_slots=None, ext_channels=None):
        super(Transformer, self).__init__()
        self.ext_flag = ext
        self.relu = nn.ReLU()
        self.in_conv = nn.Conv2d(in_channels, hidden_channels, 1)
        channels = [in_channels] + [hidden_channels] * n_layers
        self.n_pred = n_pred

        layers = []
        for i in range(n_layers):
            layers += [Attention(num_nodes, hidden_channels, 3, stnorm_bool=stnorm_bool, snorm_bool=snorm_bool, tnorm_bool=tnorm_bool, device=device)]
        self.layers = nn.ModuleList(layers)
        # self.out_conv = nn.Conv2d(hidden_channels, n_pred, 1)
        self.out_conv = nn.Conv2d(hidden_channels, out_channels, 1)

    def forward(self, x):
        b, t, n, ic = x.size()
        x = x.permute(0, 3, 2, 1)

        x_in = self.in_conv(x)
        x_out = []
        for timestep in range(self.n_pred):     # stepwise
            x_in_t = torch.cat([x_in[...,timestep:]] + x_out[timestep:], dim=-1)
            for i, layer in enumerate(self.layers):
                x_in_t = layer(x_in_t)
            x_out.append(x_in_t[...,-1:])
        # out = x[..., -1:]
        x_out = torch.cat(x_out, dim=-1)
        out = self.out_conv(x_out).permute(0, 3, 2, 1)
        return out

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if isinstance(param, Parameter):
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print(name)
                print(param.shape)
                
def main():
    TIMSSTEP_IN = 12
    N_NODE = 8
    CHANNEL = 1
    GPU = sys.argv[-1] if len(sys.argv) == 2 else '3'
    device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
    model = Transformer(num_nodes=8, in_channels=1, n_his=12, n_pred=12, hidden_channels=64, n_layers=3, device=device).to(device)
    summary(model, (12, 8, 1), device=device)
    
if __name__ == '__main__':
    main()