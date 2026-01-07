import torch.nn as nn
from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor
import torch
import math
from SignalEmbedding import *
from TransformerEncoderlayer import *
from info_nce import InfoNCE
from Featurelayer import FeatureLayer
from model_BC import *
from ECGElector import ECGElector
class Model_All(nn.Module):
    def __init__(self):
        super(Model_All, self).__init__()
        self.elector = ECGElector(num_selected=6, group_size=60)
        self.linear = nn.Linear(512,1)
        self.model = Model()
        self.scorer = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1) 
        )
        self.dropout1 = nn.Dropout(0.1)
        self.topk_num = 3
        #GateAttention
        self.attV = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh()
        )
        self.attU = nn.Sequential(
            nn.Linear(512, 128),
            nn.Sigmoid()
        )
        self.attW = nn.Linear(128, 1)

        self.aggregation_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, batch_first=True),
            num_layers=2
        )
        
    def forward(self, x_list: Tensor, positional_list: Tensor,data_mask_list: Tensor, patientinfo: Tensor, timeinfo_list: Tensor, chara_list: Tensor, bag_mask:Tensor, bc_only: bool=True) -> Tensor:

        device = x_list.device

        selected_idx = self.elector(x_list, data_mask_list, bag_mask)

        batch_size, bag_size, *rest = x_list.shape
        selected_idx = torch.clamp(selected_idx.long(), 0, bag_size - 1)
        
        batch_idx = torch.arange(batch_size, device=device).view(-1, 1).expand_as(selected_idx)
        
        x_list = x_list[batch_idx, selected_idx]
        chara_list = chara_list[batch_idx, selected_idx]
        timeinfo_list = timeinfo_list[batch_idx, selected_idx]
        positional_list = positional_list[batch_idx, selected_idx]
        data_mask_list = data_mask_list[batch_idx, selected_idx]

        B, N, *rest = x_list.shape
        x = x_list.view(B * N, *rest)
        B, N, H, D = positional_list.shape

        positional = positional_list.view(B * N, H, D)
        data_mask = data_mask_list.view(B * N, H)
        B, N, *rest = timeinfo_list.shape
        timeinfo = timeinfo_list.view(B * N, *rest)
        B, N, *rest = chara_list.shape
        chara = chara_list.view(B * N, *rest)
        #
        _, *rest = patientinfo.shape
        patientinfo_expanded = patientinfo.unsqueeze(1).expand(-1, N, -1, -1).contiguous().view(B * N, *rest)

        out, contrastive_loss, ca_out, resnet_embed_full, feature_embed_full = \
            self.model(x, positional, data_mask, patientinfo_expanded, timeinfo, chara, bc_only)

        #
        out = out.view(B, N, -1)

        #
        ca_out = ca_out.view(B, N, -1)
        resnet_embed_full = resnet_embed_full.view(B, N, -1)
        feature_embed_full = feature_embed_full.view(B, N, -1)
        
        a_v = self.attV(out)  
        a_u = self.attU(out)  
        scores = self.attW(a_v * a_u)
        ###
        weights = F.softmax(scores, dim=1) 
        out = weights * out

        out = self.aggregation_transformer(out)
        out = out.mean(dim=1)
        out = self.linear(out)
        if bc_only:
            return torch.sigmoid(out)
        else:
            return out,None, ca_out, resnet_embed_full, feature_embed_full

