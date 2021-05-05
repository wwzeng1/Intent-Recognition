import numpy as np
import pickle
import os
seqs = []
for root,_,files in os.walk("../Data/"):
    for filename in files:
        with open(os.path.join(root,filename),"rb") as f:
            train = pickle.load(f)
            for k,v in train.items():
                seqs.append(v)
print(np.concatenate(seqs))
np.save("raw_corpus.npy",np.concatenate(seqs),allow_pickle=True)
# with open("raw_corpus.tsv","w",encoding="utf-8") as fp:
#     sentences = [(" ").join(x) for x in seqs]
#     fp.write(("\n").join(sentences))