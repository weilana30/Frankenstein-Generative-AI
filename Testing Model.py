import torch
import torch.nn as nn


import torch
import torch.nn as nn

# Re-define the CharacterLSTM class
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
class CharacterLSTM(nn.Module):
    def __init__(self):
        super(CharacterLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=48)
        self.lstm = nn.LSTM(input_size=48, hidden_size=96, batch_first=True)
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



lstm_model = CharacterLSTM()
lstm_model.load_state_dict(torch.load("frankenstein_generative_AI.pth"))


starting_prompt = ("my name is andrew ")
tokenized_starting_prompt = list(starting_prompt)

tokenized_id_prompt = torch.tensor([c2ix[character] for character in tokenized_starting_prompt]).unsqueeze(0)

lstm_model.eval()

characters_generated = 500

with torch.no_grad():
    states = lstm_model.__init_state__(1)
    for char in range(characters_generated):
        output, states = lstm_model.forward(tokenized_id_prompt, states)
        predicted_id = torch.argmax(output[-1, :], dim=-1).item()
        predicted_character = ix2c[predicted_id]
        starting_prompt += predicted_character
        tokenized_id_prompt = torch.tensor([[predicted_id]])


print(starting_prompt)

