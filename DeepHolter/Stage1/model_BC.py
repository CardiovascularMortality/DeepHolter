import torch.nn as nn
from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor
import torch
import math
from SignalEmbedding import *
from TransformerEncoderlayer import *
from info_nce import InfoNCE
from Featurelayer import FeatureLayer
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embed_size = 512
        self.pos_size = 24
        self.feature_size = 1656
        self.resnet = ResNet(in_channels=12,embed_size=self.embed_size)
        self.featurelayer = FeatureLayer(self.feature_size,self.embed_size, 4, 12, self.embed_size*4, 0.1, self.pos_size)#transformer
        self.transformerencoder = Transformer(self.embed_size, 8, 6, self.embed_size*4, 0.1, self.pos_size)

        self.patientinfo_layer = nn.Sequential(
            nn.Linear(3+24, 1656),
            nn.LayerNorm(1656)
        )
        self.attn_embed = nn.Linear(self.embed_size, self.embed_size)
        self.contrastive_loss = InfoNCE(temperature=0.05)
        self.chara_bn = nn.BatchNorm1d(1656)
        # self.feature_bc = nn.Linear(self.feature_size, 1)
    
    def forward(self, x: Tensor, positional: Tensor,data_mask: Tensor, patientinfo: Tensor, timeinfo: Tensor, chara: Tensor,bc_only: bool=True) -> Tensor:
        '''
        x是含n个signal的bag
        '''
        #(batch_size, bag_size, 12, 2000)
        # print(x.shape)
        device = x.device
        batch_size, bag_size, _, _ = x.shape
        x = x.contiguous().view(-1,12,2000)#(batch_size*bag_size, 12, 2000)
        mask_expanded = data_mask.contiguous().view(-1)  # (batch_size*bag_size,)
        # print(x.shape)
        x_valid = x[mask_expanded == 1]
        # print(x_valid.shape)
        resnet_embed = self.resnet(x_valid)
        # print(resnet_embed.shape)
        resnet_embed_full = torch.zeros((x.shape[0],resnet_embed.shape[1])).contiguous().to(device)  
        resnet_embed_full[mask_expanded == 1] = resnet_embed  

        resnet_embed_full = resnet_embed_full.view(batch_size, bag_size, -1)

        positional_cls = torch.zeros((batch_size, 1, self.pos_size)).to(device)
        positional_cls[:,:,-1] = 1
        # print(positional_cls.shape)
        positional = torch.cat((positional_cls, positional), dim=1)

        mask = torch.ones((batch_size, 1)).to(device)
        data_mask = torch.cat((mask, data_mask), dim=1)

        chara = chara.contiguous().view(-1,chara.shape[-1])
        chara_valid = chara[mask_expanded==1]
        chara_valid = self.chara_bn(chara_valid)
        chara[mask_expanded == 1] = chara_valid
        chara = chara.view(batch_size, bag_size, -1)

        ##
        patientinfo = torch.cat((timeinfo,patientinfo),dim=2)
        patientinfo = self.patientinfo_layer(patientinfo) 
        feature = chara + patientinfo

        feature_embed_full = self.featurelayer(feature,data_mask,positional)


        x = self.transformerencoder(resnet_embed_full,data_mask,positional,feature_embed_full)

        ## bc classification
        # patientinfo_embed = self.patientinfo_layer(patientinfo[:,0])
        # bc_in = torch.cat((x[:,0],patientinfo_embed),dim=1)
        bc_in = x[:,0]
        # bc = self.fc_binaryClassification(bc_in)
        # bc = self.fc_binaryClassification(x[:,0])
        # print(x.shape)
        # print(x.shape)

        ##
        attn_embed = x[:,1:]
        attn_embed = attn_embed.contiguous().view(-1, self.embed_size)
        attn_embed = attn_embed[mask_expanded == 1]

        contrastive_loss = 0
        return bc_in,contrastive_loss, x, resnet_embed_full, feature_embed_full