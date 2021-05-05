
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import copy
import config
from model import *
from cm_dataloader import PhonesDataset,phoneVocab
import json
from optim import ScheduledOptim
# Train the model

def train_model(train_loader, model):
    training_loss = 0
    
    # Set model in 'Training mode'
    model.train()
    num_batches = 0
    # enumerate mini batches
    for i, (X, Y, src_pad_mask) in enumerate(train_loader):
        optimizer.zero_grad()
        X = X.to(device)
        Y = Y.to(device)
        src_pad_mask = src_pad_mask.to(device)
        
        out = model(X,src_pad_mask) # shape of out N,S,E
        # out = out.permute(1,0,2)
        loss = criterion(out.view(-1,nclasses),Y.view(-1))

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step_and_update_lr()

        training_loss += loss.item()
        # out = out.detach().float().cpu().numpy()
        # Y=Y.cpu()
        #ldist += calc_ldist(out,Y)

        torch.cuda.empty_cache()
        # print("backprop done")
        num_batches += 1

        if i % 10 == 0: 
            print("Training {}, Loss: {}".format(i, loss.item()))

    training_loss /= num_batches
    return training_loss


def evaluate_model(val_loader, model):
    val_loss=0
    # Set model in validation mode
    num_batches=0
    model.eval()
    print("Started validating")
    with torch.no_grad():
        for i, (X, Y,src_pad_mask) in enumerate(val_loader):
        
            X = X.to(device)
            Y = Y.to(device)
            src_pad_mask = src_pad_mask.to(device)
            
            out = model(X,src_pad_mask) # shape of out N,S,E
            # out = out.permute(1,0,2)
            loss = criterion(out.view(-1,nclasses),Y.view(-1))
            val_loss += loss.item()
            num_batches+=1
        
    val_loss /= num_batches
    return val_loss
 
 # Define number of epochs

def get_data():
    # Training data
    X = np.load('drive/MyDrive/11785/Project/dutch_train_data.npy', allow_pickle=True)   
    Y = np.load('drive/MyDrive/11785/Project/dutch_train_labels.npy', allow_pickle=True)
    validation_split = 0.9

    n = len(X)
    data_labels = [(X[i], Y[i]) for i in range(n)]

    indices = list(range(n))
    val_split = int(np.floor(validation_split * n))

    train_data, val_data, = random_split(data_labels, [val_split, n - val_split])
    return train_data, val_data
    


if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    num_workers = 8 if cuda else 0
    print("Cuda = "+str(cuda)+" with num_workers = "+str(num_workers))
    device = torch.device("cuda" if cuda else "cpu")
    
    #get data
    train_data, val_data = get_data()

    ## Dataloaders
    with open(config.vocab_file_path,"r") as f:
        vocab_map = json.load(f)
    vocab = phoneVocab(vocab_map)    
    train_data = PhonesDataset(data = train_data, vocab)
    train_loader = DataLoader(train_data, collate_fn=PhonesDataset.my_collate, shuffle=True, batch_size=batch_size)

    # Validation dataloader tbd

    # Model
    torch.manual_seed(11785)
    torch.backends.cudnn.deterministic = True
    vocab_size = vocab.vocab_size
    em_size = config.em_size
    num_heads = config.num_heads
    hid_dim = config.hid_dim
    nencoderlayers = config.nencoderlayers
    dout = config.dout
    nclasses = 4
    model = Model( num_classes=nclasses,vocab_size=vocab_size,embed_size=em_size,nheads=num_heads,num_encoder_layers=nencoderlayers,hidden_size=hid_dim,dropout=dout)

    print(model)
    #xavier uniform initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    model=model.to(device) #.cuda()
    # model.load_state_dict(torch.load("v0.pt"))
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    optimizer = ScheduledOptim(
        torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        em_size, n_warmup_steps=config.WARMUP)

    epochs = config.EPOCHS
    minloss = sys.maxsize

    for epoch in range(epochs):
        training_loss = train_model(train_loader, model)
        #val_loss = evaluate_model(val_loader, model)

        if training_loss < minloss:
        # if best_accuracy > val_loss:
            # best_accuracy = val_loss
            minloss= training_loss
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(), "v0.pt")
            del best_model
        # Print log of accuracy and loss
        print("Epoch: "+str(epoch)+", Training loss: "+str(training_loss))#+", Validation loss:"+str(val_loss)+", Levenstein distance:"+str(ldist)+"\n")
            # ", Validation accuracy:"+str(val_acc*100)+"%")
