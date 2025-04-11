import torch
from torch import nn


class LSTM_Model(nn.Module):
    def __init__(self, vocab_size, glove_dim, num_lstm_layers, hidden_size):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, glove_dim)
        self.lstm = nn.LSTM(glove_dim, hidden_size = hidden_size, num_layers = num_lstm_layers, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(hidden_size * 2 , hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        out = self.embedding(x)
        out, (h, c) = self.lstm(out)
        out = out[:, -1, :]
        out = self.dropout(self.relu(self.fc1(out)))
        out = self.fc2(out)
        return out