import numpy as np
import sentencepiece as spm
import torch
from torch.utils.data import DataLoader, Dataset
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


class LMDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, vocab, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.vocab = vocab
        self.num_batches = len(self.dataset) // self.batch_size
        
    def __iter__(self):
        
        batches_x=[]
        batches_y=[]
        idx=0
        while idx < len(self.dataset):
            p = np.random.random_sample()
            if p < 0.95:
                seqLen = round(np.random.normal(80,10))
            else:
               seqLen = round(np.random.normal(20, 5))
            if idx+(seqLen*self.batch_size) <= len(self.dataset):
                t = self.dataset[idx:idx+(seqLen*self.batch_size)].reshape(self.batch_size,-1)
            else:
                break
            inputs = []
            targets = []
            for i in range(t.shape[0]):
                t1_random, t1_label = self.mask_random_tokens(t[i])
                # t1_label = np.pad(t1_label, (0, seqLen - len(t1_label)), mode='constant')
                t1_random = np.hstack(([self.vocab.sos_id],t1_random,[self.vocab.eos_id])).astype(int)
                t1_label = np.hstack(([self.vocab.pad_id],t1_label,[self.vocab.pad_id]))
                inputs.append(t1_random)
                targets.append(t1_label)
            inputs = torch.LongTensor(np.stack(inputs))
            targets = torch.LongTensor(np.stack(targets))
            batches_x.append(inputs)
            batches_y.append(targets)
            idx += seqLen*self.batch_size
        indices = np.arange(len(batches_y))
        if self.shuffle:
            np.random.shuffle(indices)
        for i in indices:
            yield batches_x[i],batches_y[i]

    def mask_random_tokens(self, chars):
        # tokens = sentence.split()
        # tokens_len = [len(token) for token in tokens]
        # chars = [char for char in sentence]
        output_label = []

        for i, char in enumerate(chars):
            prob = np.random.random_sample()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    chars[i] = self.vocab.mask_id

                # 10% randomly change token to random token
                elif prob < 0.9:
                    chars[i] = np.random.randint(0,self.vocab.vocab_size)

                # 10% randomly change token to current token
                else:
                    chars[i] = self.vocab.ph2id(char)

                output_label.append(self.vocab.ph2id(char))

            else:
                chars[i] = self.vocab.ph2id(char)
                output_label.append(0)

        return chars, output_label

        
