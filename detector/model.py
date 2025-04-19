import torch
from torch import nn
import math


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
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerEncoder_Model(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        # self.glove_dim = glove_dim
        self.embedding = nn.Embedding(vocab_size, d_model)
        # self.embedding.weight.data.copy_(torch.from_numpy(glove_weights))
        # self.embedding.weight.requires_grad = False
        self.pos_encoder = PositionalEncoding(d_model, 100)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x, src_key_padding_mask=None):
        out = self.embedding(x)
        out = self.pos_encoder(out)
        out = self.transformer_encoder(out, src_key_padding_mask=src_key_padding_mask)
        out = out.mean(dim=1)
        out = self.fc(out)
        return out