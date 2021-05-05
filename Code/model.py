
import os
import sys
import numpy as np
#import pandas as pd
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import copy
import math
## Model Architecture definition

class Model(nn.Module):
    def __init__(self,vocab_size, embed_size, nheads,num_encoder_layers,hidden_size,dropout=0.15):
        super(Model, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_size,dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nheads, hidden_size, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        # self.kernel_size1 = 3
        # self.kernel_size2 = 3
        # self.stride1 = 2
        # self.stride2 = 2
        # self.embed_src = nn.Sequential(nn.Conv1d(40,64,kernel_size=self.kernel_size1,stride=self.stride1),nn.BatchNorm1d(64),nn.ReLU(),
        #                     nn.Conv1d(64,embed_size,kernel_size=self.kernel_size2,stride=self.stride1),nn.BatchNorm1d(embed_size),nn.ReLU())
                            # nn.Conv1d(embed_size,embed_size,kernel_size=self.kernel_size2,stride=self.stride2),nn.BatchNorm1d(embed_size),nn.ReLU())
        self.embed_size = embed_size
        self.embed_src = nn.Embedding(vocab_size,self.embed_size)
        self.linear = nn.Linear(self.embed_size,self.embed_size)
        self.decoder = nn.Sequential(nn.Linear(self.embed_size,self.embed_size),nn.ReLU(),nn.Linear(self.embed_size, vocab_size))
        self.softmax = nn.LogSoftmax(dim=-1)
        #weight tying
        # self.decoder[2].weight = self.embed_src.weight
        # self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.embed_src(src) * math.sqrt(self.embed_size)
        # src = self.embed_src(src.permute(0,2,1))
        #out shape from conv N,C,L
        # src = src.permute(2,0,1) * math.sqrt(self.embed_size)
        #expected in shape to transformer S,N,E
        src = self.pos_encoder(src.permute(1,0,2))

        output = self.transformer_encoder(src,src_key_padding_mask=None)
        #output of transformer S,N,E
        output = output.permute(1,0,2)
        #expexted nn Linear shape N,*,E
        output = self.decoder(output)
        output = self.softmax(output)
        # print(output.shape)
        return output
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)