import torch
import torch.nn as nn 
from torch import embedding, tensor as kelechi
from config import Config
from torch.nn.utils.rnn import pack_padded_sequence

class TextRNNDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, embed_dim, num_layers, max_len=20):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm =  nn.LSTM(
            embed_dim, hidden_size, num_layers
        )
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_len = max_len

    def forward(self, features, captions, lengths):
        text_embed = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), text_embed), dim=2)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hidden, _ = self.lstm(packed)
        rnn_output = self.linear(hidden[0])

        return rnn_output
    
    def sample(self, features, states=None):
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
    

            


# class Attention(nn.Module):
#     def __init__(self, model_dim, n_heads=8, dropout=0.0, project_dropout=0.0):
#         super().__init__()
#         self.num_heads = n_heads
#         self.scale = 1.0 / model_dim**0.5

#         self.qkv = nn.Linear(model_dim, model_dim * 3, bias=False)
#         self.attend_dropout = nn.Dropout(dropout)
#         self.output_layer = nn.Sequential(
#             nn.Linear(model_dim, model_dim), nn.Dropout(project_dropout)
#         )

#     def forward(self, x):
#         a, b, c = x.shape
#         qkv = self.qkv(x).reshape(a, b, 3, self.num_heads, c // self.num_heads)
#         q, k, v = qkv.permute(2, 0, 3, 1, 4)

#         dot_prod = (q @ k.transpose(-2, -1)) * self.scale
#         attention_score_nodrop = dot_prod.softmax(dim=1)
#         attention_score = self.attend_dropout(attention_score_nodrop)

#         x = (attention_score @ v).transpose(1, 2).reshape(a, b, c)
#         x = self.output_layer(x)

#         return x
