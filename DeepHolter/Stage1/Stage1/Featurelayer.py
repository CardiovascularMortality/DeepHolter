import torch.nn as nn
from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor
import torch
from TransformerEncoderlayer import *

class FeatureLayer(nn.Module):
    def __init__(self, d_in, d_model, num_heads, num_layers, d_ff, dropout, max_seq_length):
        '''
        d_model: The dimensionality of the input.
        num_heads: The number of attention heads in the multi-head attention.
        num_layers: The number of encoder layers.
        d_ff: The dimensionality of the inner layer in the position-wise feed-forward network.
        max_seq_length: The maximum sequence length.
        dropout: The dropout rate used for regularization.
        '''
        super(FeatureLayer, self).__init__()

        self.positional_encoding = BiphasePositionalEncoding(d_in, max_seq_length)
        self.num_heads = num_heads

        self.signal_cls_token = nn.Parameter(torch.randn(1, 1, d_in))

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_in, num_heads, d_in*4, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout1d(dropout)
        self.fc3 = nn.Linear(d_in, d_model)
        self.bn3 = nn.LayerNorm(d_model)
    def forward(self, x, mask, hour_onehot):
        '''
        x: The input tensor of shape (batch_size, seq_len, d_model). x is the embedding trained from Resnet34(signal) or Linear(feature).
        '''


        expand_cls_token = self.signal_cls_token.expand(x.size(0), -1,-1)  
        x = torch.cat((expand_cls_token, x), dim=1)  

        x = self.dropout(self.positional_encoding(x,hour_onehot))
        
        mask = (mask==0)
        mask = mask.view(mask.size(0), 1, 1, mask.size(1)).expand(-1, self.num_heads, -1, -1)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x,x,x,mask)

        x = self.bn3(self.fc3(x))
        return x