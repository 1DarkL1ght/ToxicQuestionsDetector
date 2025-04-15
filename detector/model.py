import torch
from torch import nn


class LSTMAttention_Model(nn.Module):
    def __init__(self, vocab_size, glove_dim, num_lstm_layers, hidden_size):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, glove_dim)
        self.lstm = nn.LSTM(glove_dim, hidden_size = hidden_size, num_layers = num_lstm_layers, bidirectional=True, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_size * 2, 8, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size * 2 , 1)
    
    def forward(self, x):
        out = self.embedding(x)
        out, (h, c) = self.lstm(out)
        query = out[:, -1, :].unsqueeze(1)
        key = out
        value = out
        attn_out, attn_weights = self.attention(query, key, value)
        out = attn_out.flatten(start_dim=1)
        out = self.fc1(out)
        return out