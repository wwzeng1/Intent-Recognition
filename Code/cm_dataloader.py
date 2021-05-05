import os
import sys
import numpy as np
#import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import copy
import config
import json
class phoneVocab():
    def __init__(self,vocab_map):
        self.mask_id = 1
        self.pad_id = 0
        self.eos_id = 3
        self.sos_id = 2
        self.vocab_size = len(vocab_map)
        self.mapping = vocab_map
        self.inv_mapping = None
    def ph2id(self,phone):
        return self.mapping[phone]
    def id2ph(self,indx):
        return self.inv_mapping[indx]
# Dataset definition
class PhonesDataset(Dataset):
    # load the dataset
    def __init__(self, data, vocab, mode="Train"):
        self.X = [data for (data, labels) in data]
        self.Y = [labels for (data, labels) in data]
        self.vocab = vocab
        if mode == "test": 
            # forgot to do the conversions, upload these later 
            self.X = np.load('./test.npy', allow_pickle=True)   
        
            self.X = [torch.from_numpy(x) for x in self.X]
            self.Y = None

    def __len__(self):
        return len(self.X)
    # number of rows in the dataset
    def __len__(self):
        return len(self.y)

    # get a row at an index
    def __getitem__(self, index):
        x = [self.vocab.sos_id]+ [self.vocab.ph2id(phone) for phone in self.X[index]]+[self.vocab.eos_id]
        x = torch.LongTensor(x)
        y = torch.Tensor(self.Y[index])
        return x,y

    def my_collate(batch):

        X = [item[0] for item in batch]
        Y = [item[1] for item in batch]
        # target = torch.LongTensor(target)
        X_lens = [len(seq) for seq in X]
        max_x_len = max(X_lens)

        src_masks = []
        for i in range(len(batch)):
            # Generate the masks for source and target, False if there's a token, True if there's padding
            src_masks.append([False for _ in range(X_lens[i])] + [True for _ in range(max_x_len - X_lens[i])])
            # Add 0 padding
            to_pad_x = torch.LongTensor([self.vocab.pad_id for _ in range(max_x_len - X_lens[i])])
            X[i] = torch.cat((X[i],to_pad_x),dim=0)

        return torch.stack(X,dim=0),torch.stack(Y,dim=0),torch.BoolTensor(src_masks),
