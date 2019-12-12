import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import time

import shakespeare_data as sh

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

corpus=sh.read_corpus()
chars, charmap = sh.get_charmap(corpus)


class TextDataset(Dataset):
    def __init__(self,text, seq_len=200):
        n_seq=len(text)//seq_len
        text=text[:n_seq*seq_len]
        self.data=torch.tensor(text).view(-1,seq_len)

    def __getitem__(self,i):
        txt=self.data[i]
        return txt[:-1],txt[1:]

    def __len__(self):
        return self.data.size(0)

def collate(seq_list):
    inputs=torch.cat([s[0].unsqueeze(1) for s in seq_list],dim=1)
    targets=torch.cat([s[1].unsequeeze(1) for s in seq_list],dim=1)
    return inputs, targets

class CharLanguageModel(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden_size,nlayers):
        super(CharLanguageModel,self).__init__()
        self.vocab_size=vocab_size
        self.embed_size=embed_size
        self.hidden_size=hidden_size
        self.nlayers=nlayers
        self.embedding=nn.Embedding(vocab_size,embed_size)
        self.rnn=nn.LSTM(input_size=embed_size,hidden_size=hidden_size,num_layers=nlayers)
        self.scoring=nn.Linear(hidden_size,vocab_size)

    def forward(self,seq_batch):
        batch_size=seq_batch.size(1)
        embed=self.embedding(seq_batch)
        hidden=None
        output_lstm,hidden=self.rnn(embed,hidden)
        output_lstm_flatten=output_lstm.view(-1,self.hidden_size)
        output_flatten=self.scoring(output_lstm_flatten)
        return output_flatten.view(-1,batch_size,self.vocab_size)

    def generate(self,seq, n_words): # L x V
        # performs greedy search to extract and return words (one sequence).
        generated_words = []
        embed = self.embedding(seq).unsqueeze(1) # L x 1 x E
        hidden = None
        output_lstm, hidden = self.rnn(embed,hidden) # L x 1 x H
        output = output_lstm[-1] # 1 x H
        scores = self.scoring(output) # 1 x V
        _,current_word = torch.max(scores,dim=1) # 1 x 1
        generated_words.append(current_word)
        if n_words > 1:
            for i in range(n_words-1):
                embed = self.embedding(current_word).unsqueeze(0) # 1 x 1 x E
                output_lstm, hidden = self.rnn(embed,hidden) # 1 x 1 x H
                output = output_lstm[0] # 1 x H
                scores = self.scoring(output) # V
                _,current_word = torch.max(scores,dim=1) # 1
                generated_words.append(current_word)
        return torch.cat(generated_words,dim=0)


def train_epoch(model, optimizer, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(DEVICE)
    before = time.time()
    print("training", len(train_loader), "number of batches")
    for batch_idx, (inputs,targets) in enumerate(train_loader):
        if batch_idx == 0:
            first_time = time.time()
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        outputs = model(inputs) # 3D
        loss = criterion(outputs.view(-1,outputs.size(2)),targets.view(-1)) # Loss of the flattened outputs
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx == 0:
            print("Time elapsed", time.time() - first_time)

        if batch_idx % 100 == 0 and batch_idx != 0:
            after = time.time()
            print("Time: ", after - before)
            print("Loss per word: ", loss.item() / batch_idx)
            print("Perplexity: ", np.exp(loss.item() / batch_idx))
            after = before

    val_loss = 0
    batch_id=0
    for inputs,targets in val_loader:
        batch_id+=1
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        outputs = model(inputs)
        loss = criterion(outputs.view(-1,outputs.size(2)),targets.view(-1))
        val_loss+=loss.item()
    val_lpw = val_loss / batch_id
    print("\nValidation loss per word:",val_lpw)
    print("Validation perplexity :",np.exp(val_lpw),"\n")
    return val_lpw

model = CharLanguageModel(charcount,256,256,3)
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=1e-6)
split = 5000000
train_dataset = TextDataset(shakespeare_array[:split])
val_dataset = TextDataset(shakespeare_array[split:])
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=64, collate_fn = collate)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=64, collate_fn = collate, drop_last=True)

for i in range(3):
    train_epoch(model, optimizer, train_loader, val_loader)

def generate(model, seed,nwords):
    seq = sh.map_corpus(seed, charmap)
    seq = torch.tensor(seq).to(DEVICE)
    out = model.generate(seq,nwords)
    return sh.to_text(out.cpu().detach().numpy(),chars)

print(generate(model, "Richard ", 1000))

stop_character = charmap['\n']
space_character = charmap[" "]
lines = np.split(shakespeare_array, np.where(shakespeare_array == stop_character)[0]+1) # split the data in lines
shakespeare_lines = []
for s in lines:
    s_trimmed = np.trim_zeros(s-space_character)+space_character # remove space-only lines
    if len(s_trimmed)>1:
        shakespeare_lines.append(s)
for i in range(10):
    print(sh.to_text(shakespeare_lines[i],chars))
print(len(shakespeare_lines))

class LinesDataset(Dataset):
    def __init__(self,lines):
        self.lines=[torch.tensor(l) for l in lines]
    def __getitem__(self,i):
        line = self.lines[i]
        return line[:-1].to(DEVICE),line[1:].to(DEVICE)
    def __len__(self):
        return len(self.lines)

# collate fn lets you control the return value of each batch
# for packed_seqs, you want to return your data sorted by length
def collate_lines(seq_list):
    inputs,targets = zip(*seq_list)
    lens = [len(seq) for seq in inputs]
    seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
    inputs = [inputs[i] for i in seq_order]
    targets = [targets[i] for i in seq_order]
    return inputs,targets

class PackedLanguageModel(nn.Module):

    def __init__(self,vocab_size,embed_size,hidden_size, nlayers, stop):
        super(PackedLanguageModel,self).__init__()
        self.vocab_size=vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.nlayers=nlayers
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.rnn = nn.LSTM(input_size = embed_size,hidden_size=hidden_size,num_layers=nlayers) # 1 layer, batch_size = False
        self.scoring = nn.Linear(hidden_size,vocab_size)
        self.stop = stop # stop line character (\n)

    def forward(self,seq_list): # list
        batch_size = len(seq_list)
        lens = [len(s) for s in seq_list] # lens of all lines (already sorted)
        bounds = [0]
        for l in lens:
            bounds.append(bounds[-1]+l) # bounds of all lines in the concatenated sequence. Indexing into the list to
                                        # see where the sequence occurs. Need this at line marked **
        seq_concat = torch.cat(seq_list) # concatenated sequence
        embed_concat = self.embedding(seq_concat) # concatenated embeddings
        embed_list = [embed_concat[bounds[i]:bounds[i+1]] for i in range(batch_size)] # embeddings per line **
        packed_input = rnn.pack_sequence(embed_list) # packed version

        # alternatively, you could use rnn.pad_sequence, followed by rnn.pack_padded_sequence



        hidden = None
        output_packed,hidden = self.rnn(packed_input,hidden)
        output_padded, _ = rnn.pad_packed_sequence(output_packed) # unpacked output (padded). Also gives you the lengths
        output_flatten = torch.cat([output_padded[:lens[i],i] for i in range(batch_size)]) # concatenated output
        scores_flatten = self.scoring(output_flatten) # concatenated logits
        return scores_flatten # return concatenated logits

    def generate(self,seq, n_words): # L x V
        generated_words = []
        embed = self.embedding(seq).unsqueeze(1) # L x 1 x E
        hidden = None
        output_lstm, hidden = self.rnn(embed,hidden) # L x 1 x H
        output = output_lstm[-1] # 1 x H
        scores = self.scoring(output) # 1 x V
        _,current_word = torch.max(scores,dim=1) # 1 x 1
        generated_words.append(current_word)
        if n_words > 1:
            for i in range(n_words-1):
                embed = self.embedding(current_word).unsqueeze(0) # 1 x 1 x E
                output_lstm, hidden = self.rnn(embed,hidden) # 1 x 1 x H
                output = output_lstm[0] # 1 x H
                scores = self.scoring(output) # V
                _,current_word = torch.max(scores,dim=1) # 1
                generated_words.append(current_word)
                if current_word[0].item()==self.stop: # If end of line
                    break
        return torch.cat(generated_words,dim=0)

def train_epoch_packed(model, optimizer, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss(reduction="sum") # sum instead of averaging, to take into account the different lengths
    criterion = criterion.to(DEVICE)
    batch_id=0
    before = time.time()
    print("Training", len(train_loader), "number of batches")
    for inputs,targets in train_loader: # lists, presorted, preloaded on GPU
        batch_id+=1
        outputs = model(inputs)
        loss = criterion(outputs,torch.cat(targets)) # criterion of the concatenated output
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_id % 100 == 0:
            after = time.time()
            nwords = np.sum(np.array([len(l) for l in inputs]))
            lpw = loss.item() / nwords
            print("Time elapsed: ", after - before)
            print("At batch",batch_id)
            print("Training loss per word:",lpw)
            print("Training perplexity :",np.exp(lpw))
            before = after
    
    val_loss = 0
    batch_id=0
    nwords = 0
    for inputs,targets in val_loader:
        nwords += np.sum(np.array([len(l) for l in inputs]))
        batch_id+=1
        outputs = model(inputs)
        loss = criterion(outputs,torch.cat(targets))
        val_loss+=loss.item()
    val_lpw = val_loss / nwords
    print("\nValidation loss per word:",val_lpw)
    print("Validation perplexity :",np.exp(val_lpw),"\n")
    return val_lpw

model = PackedLanguageModel(charcount,256,256,3, stop=stop_character)
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=1e-6)
split = 100000
train_dataset = LinesDataset(shakespeare_lines[:split])
val_dataset = LinesDataset(shakespeare_lines[split:])
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=64, collate_fn = collate_lines)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=64, collate_fn = collate_lines, drop_last=True)

for i in range(20):
    train_epoch_packed(model, optimizer, train_loader, val_loader)

torch.save(model, "trained_model.pt")

print(generate(model, "Richard ", 1000))


