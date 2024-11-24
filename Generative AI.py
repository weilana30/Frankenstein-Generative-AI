import numpy as np
import torch
import torch.nn as nn

##opening text file
with open("Frankenstein.txt", 'r') as file:
    original_text = file.read()


#getting text in list and removing weird characters
original_text = original_text.lower()

tokenized_text = list(original_text)

unique_characters = sorted(list(set(tokenized_text)))
##indexing characters
c2ix = {character: ix for ix, character in enumerate(unique_characters)}

ix2c = {ix: character for character, ix in c2ix.items()}

vocab_size = len(c2ix)
print(vocab_size)

tokenized_id_text = [c2ix[word] for word in tokenized_text]

from torch.utils.data import Dataset, DataLoader


##Creates features and labels
class TextDataset(Dataset):
    def __init__(self, tokenized_text, seq_length):
        self.tokenized_text = tokenized_text
        self.seq_length = seq_length

    ##
    def __len__(self):
        return len(self.tokenized_text) - self.seq_length

    def __getitem__(self, idx):
        features = torch.tensor(self.tokenized_text[idx: idx+self.seq_length])
        labels = torch.tensor(self.tokenized_text[idx+ 1 :idx+self.seq_length + 1])
        return features, labels

seq_length = 48

dataset = TextDataset(tokenized_id_text, seq_length)

batch_size = 36

##creates an iterable to train LSTM
dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=True)

##Generates text using character based tokens
class CharacterLSTM(nn.Module):
    def __init__(self):
        super(CharacterLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=48)
        self.lstm = nn.LSTM(input_size = 48, hidden_size= 96, batch_first=True)
        self.linear = nn.Linear(96, vocab_size)


    def forward(self, x, states):
        x = self.embedding(x)
        out, states = self.lstm(x, states)
        out = self.linear(out)
        out = out.reshape(-1, out.size(2))
        return out, states

    def __init_state__(self, batch_size):
        hidden = torch.zeros(1, batch_size, 96)
        cell = torch.zeros(1, batch_size, 96)
        return hidden, cell

##Instance of CharacterLSTM()
lstm_model = CharacterLSTM()


##loss function
loss = nn.CrossEntropyLoss()

import torch.optim as optim

optimizer = optim.Adam(lstm_model.parameters(), lr = 0.015)


num_epochs = 5

iteration = 0
for epoch in range(num_epochs):
    for features, labels in dataloader:
        optimizer.zero_grad()
        states = lstm_model.__init_state__(features.size(0))
        outputs, states = lstm_model(features, states)
        CELoss = loss(outputs, labels.view(-1))
        CELoss.backward()
        optimizer.step()

        iteration += 1

        ##Print only every 100 iterations
        if iteration % 100 == 0:
            print(f"Iteration {iteration} | Epoch {epoch + 1} | Loss: {CELoss.item()}")


torch.save(lstm_model.state_dict(), "frankenstein_generative_AI.pth")
print("Model Saved")





