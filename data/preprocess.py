import csv
import numpy as np

f = open('dstc2/train/transcriptions_S0.csv', newline='', encoding = 'utf8')
with f as f:
    csvread = csv.reader(f)
    batch_data = list(csvread)
eng_phones = ['a', 'aː', 'b', 'd', 'd̠', 'e', 'eː', 'e̞', 
    'f', 'h', 'i', 'iː', 'j', 'k', 'kʰ', 'l', 'm', 'n', 'o', 
    'oː', 'p', 'pʰ', 'r', 's', 't', 'tʰ', 't̠', 'u', 'uː', 
    'v', 'w', 'x', 'z', 'æ', 'ð', 'øː', 'ŋ', 'ɐ', 'ɐː', 'ɑ', 
    'ɑː', 'ɒ', 'ɒː', 'ɔ', 'ɔː', 'ɘ', 'ə', 'əː', 'ɛ', 'ɛː', 'ɜː', 
    'ɡ', 'ɪ', 'ɪ̯', 'ɯ', 'ɵː', 'ɹ', 'ɻ', 'ʃ', 'ʉ', 'ʉː', 'ʊ', 'ʌ', 
    'ʍ', 'ʒ', 'ʔ', 'θ']
count = len(eng_phones)
eng_dict = dict()
for idx in range(count):
    phone = eng_phones[idx]
    eng_dict[phone] = idx

all_x = []
all_y = []
for line in batch_data:
    data_id = line[0]
    file_name = line[1]
    str_list = ','.join(line[2:-1])
    data = [eng_dict[i] for i in str_list if i in eng_phones]
    all_x.append(data)
    labels = line[-1]
    all_y.append(labels)
train_labels = list(set(all_y))
train_labels = sorted(train_labels)

label_dict = dict()
for idx in range(len(train_labels)):
    label = train_labels[idx]
    label_dict[label] = idx
all_y = [label_dict[i] for i in all_y]

all_x = np.array([np.array(xi) for xi in all_x], dtype=object)
all_y = np.array(all_y)

np.save('S0_train_data.npy', all_x, allow_pickle=True)
np.save('S0_train_labels.npy', all_y, allow_pickle=True)