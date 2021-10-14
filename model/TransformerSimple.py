from torch._C import device
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys 
from torchsummary import summary
import math


class Transformer(nn.Module):
    def __init__(self, device, feature=10, time_in=12, dmodel=64, nhead=8, num_layers=1):
        super(Transformer, self).__init__()
        self.device = device
        self.input_embedding = nn.Linear(feature, dmodel)
        self.temporal_encoder_layer = nn.TransformerEncoderLayer(dmodel, nhead)
        self.temporal_encoder = nn.TransformerEncoder(self.temporal_encoder_layer, num_layers)
        self.temporal_pos = torch.arange(0, time_in).to(device=device)
        self.temporal_pe = nn.Embedding(time_in, dmodel)
        self.decoder = nn.Linear(dmodel, feature)

    def forward(self, x):
        b, t, n = x.shape
        x = self.input_embedding(x)
        t_pe = self.temporal_pe(self.temporal_pos).expand(b, t, -1)
        x = x + t_pe
        x = self.temporal_encoder(x)
        x = self.decoder(x)
        return x

def print_params(model_name, model):
    param_count=0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count += param.numel()
    print(f'{model_name}, {param_count} trainable parameters in total.')
    return

def main():
    CHANNEL, N_NODE, TIMESTEP_IN = 1, 8, 12
    GPU = sys.argv[-1] if len(sys.argv) == 2 else '0'
    device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
    model = Transformer(device, feature=N_NODE*CHANNEL, time_in=TIMESTEP_IN, dmodel=64, nhead=8, num_layers=1).to(device)
    summary(model, (TIMESTEP_IN, N_NODE), device=device)
    print_params("Transformer", model)
    
if __name__ == '__main__':
    main()
