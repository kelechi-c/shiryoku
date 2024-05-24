from pprint import pprint
import torch
import torch.nn as nn 
from config import Config
from torch.nn.utils.rnn import pack_padded_sequence

class TextRNNDecoder(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=4, max_len=30):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm =  nn.LSTM(
            embed_dim, hidden_size, num_layers
        )
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_len = max_len

    def forward(self, features, captions, lengths):
        text_embed = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), text_embed), dim=1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hidden, _ = self.lstm(packed)
        rnn_output = self.linear(hidden[0])

        return rnn_output 


    def sample(self, features, state=None):
        sampled = []
        input_seq = features.unsqueeze(1)
        for _ in range(self.max_len):
            hidden, state = self.lstm(input_seq, state)
            outputs = self.linear(hidden.squeeze(1))
            _, predicted = outputs.max(1)
            sampled.append(predicted)
            input_seq = self.embed(predicted)
            input_seq = input_seq.unsqueeze(1)

        sampled = torch.stack(sampled, 1)

        return sampled


