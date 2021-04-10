import csv
import numpy as np

# this is a script to preprocess data given in a csv format of 
# id, file_name, phoneme sequence, and intent

f = open('dstc2/train/transcriptions_S0.csv', newline='', encoding = 'utf8')
with f as f:
    csvread = csv.reader(f)
    batch_data = list(csvread)

# list of english phones in allosaurus
eng_phones = ['a', 'aː', 'b', 'd', 'd̠', 'e', 'eː', 'e̞', 
    'f', 'h', 'i', 'iː', 'j', 'k', 'kʰ', 'l', 'm', 'n', 'o', 
    'oː', 'p', 'pʰ', 'r', 's', 't', 'tʰ', 't̠', 'u', 'uː', 
    'v', 'w', 'x', 'z', 'æ', 'ð', 'øː', 'ŋ', 'ɐ', 'ɐː', 'ɑ', 
    'ɑː', 'ɒ', 'ɒː', 'ɔ', 'ɔː', 'ɘ', 'ə', 'əː', 'ɛ', 'ɛː', 'ɜː', 
    'ɡ', 'ɪ', 'ɪ̯', 'ɯ', 'ɵː', 'ɹ', 'ɻ', 'ʃ', 'ʉ', 'ʉː', 'ʊ', 'ʌ', 
    'ʍ', 'ʒ', 'ʔ', 'θ']

count = len(eng_phones)
eng_dict = dict()
# maps phones to index, this way we can make them one hot vectors
for idx in range(count):
    phone = eng_phones[idx]
    eng_dict[phone] = idx

# initialize lists for data
all_x = []
all_y = []
for line in batch_data:
    data_id = line[0]
    file_name = line[1]
    # the line has values in list format, which is very messy
    str_list = ','.join(line[2:-1])
    # skips over everything in the string besides phonemes
    data = [eng_dict[i] for i in str_list if i in eng_phones]
    all_x.append(data)
    labels = line[-1]
    all_y.append(labels)

# generate list of possible labels
train_labels = list(set(all_y))
train_labels = sorted(train_labels)[1:] # removes the blank label

# maps these labels to indices
label_dict = dict()
for idx in range(len(train_labels)):
    label = train_labels[idx]
    label_dict[label] = idx

# this will remove the blank values
print(label_dict)

# all y is a list of strings
# all x is a list of phoneme sequences
x_array = []
y_array = []
for idx in range(len(all_y)):
    y_i = all_y[idx]
    x_i = all_x[idx]
    if y_i in label_dict:
        x_i_arr = np.array(x_i)
        x_array.append(x_i_arr)
        y_i_int = label_dict[y_i]
        y_array.append(y_i_int)

x_array = np.array(x_array, dtype=object)
y_array = np.array(y_array)

np.save('S0_train_data.npy', x_array, allow_pickle=True)
np.save('S0_train_labels.npy', y_array, allow_pickle=True)