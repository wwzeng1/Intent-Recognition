import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import copy
import Code.config as config
from Code.model import *
from Code.lm_dataloader import LMDataLoader,phoneVocab
import json
from Code.optim import ScheduledOptim
# Train the model

def train_model(train_loader, model):
    training_loss = 0
    
    # Set model in 'Training mode'
    model.train()
    num_batches = 0
    # enumerate mini batches
    for i, (X, Y) in enumerate(train_loader):
        optimizer.zero_grad()
        X = X.to(device)
        Y = Y.to(device)
        # X_lens = X_lens
        
        out = model(X) # shape of out N,S,E
        # out = out.permute(1,0,2)
        loss = criterion(out.view(-1,vocab_size),Y.view(-1))

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

        if i % 100 == 0: 
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
        for i, (X, Y) in enumerate(val_loader):
        
            X = X.to(device)
            Y = Y.to(device)
            # X_lens = X_lens
            
            out = model(X) # shape of out N,S,E
            # out = out.permute(1,0,2)
            loss = criterion(out.view(-1,vocab_size),Y.view(-1))
            val_loss += loss.item()
            num_batches+=1
        
    val_loss /= num_batches
    return val_loss
 
 # Define number of epochs

def get_data():
    # Training data
    x_train = np.load("../data/train_clean.npy", allow_pickle=True)#[:100]
    print("Shape of training data:", x_train.shape)

    # Training labels
    labels_train = np.load("../data/train_transcripts_clean.npy", allow_pickle=True)#[:100]
    print("Shape of training data labels:", labels_train.shape)

    # Validation data
    x_val = np.load("../data/dev_clean.npy", allow_pickle=True)#[:100]
    print("Shape of Validation data:", x_val.shape)
    # Validation labels
    labels_val = np.load("../data/dev_transcripts_clean.npy", allow_pickle=True)#[:100]
    # labels_val=np.concatenate(labels_val,axis=0)
    print("Shape of validation data labels", labels_val.shape)

    return x_train,x_val, labels_train, labels_val
    


if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    num_workers = 8 if cuda else 0
    print("Cuda = "+str(cuda)+" with num_workers = "+str(num_workers))
    device = torch.device("cuda" if cuda else "cpu")

    ## Dataloaders
    with open('Code/'+config.vocab_file_path,"r") as f:
        vocab_map = json.load(f)
    vocab = phoneVocab(vocab_map)    
    raw_corpus = np.load("Code/raw_corpus.npy",allow_pickle=True)
    train_loader = LMDataLoader(dataset=raw_corpus, batch_size=config.BATCH_SIZE, shuffle=True,vocab=vocab, num_workers=8)

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
    model = Model(vocab_size=vocab_size,embed_size=em_size,nheads=num_heads,num_encoder_layers=nencoderlayers,hidden_size=hid_dim,dropout=dout)

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
            torch.save(best_model.state_dict(), "drive/MyDrive/11785/Project/checkpoint/v0.pt")
            del best_model
        # Print log of accuracy and loss
        print("Epoch: "+str(epoch)+", Training loss: "+str(training_loss))#+", Validation loss:"+str(val_loss)+", Levenstein distance:"+str(ldist)+"\n")
            # ", Validation accuracy:"+str(val_acc*100)+"%")
