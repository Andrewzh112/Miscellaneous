import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, max_len, nhead, nhid, nlayers, nclasses, cat_tokens, cat_size, cat_len, dropout=0.5):
        super(TransformerModel, self).__init__()
        
        self.model_type = 'Transformer'

        # transformer layers
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)

        # embed extra categorical variables
        self.cat_embed = nn.Embedding(cat_tokens, cat_size)
        
        # final fc
        self.decoder = nn.Linear(ninp*max_len+cat_size*cat_len, nclasses)

        #dimensions
        self.ninp = ninp
        self.max_len = max_len
        self.cat_size = cat_size
        self.cat_len = cat_len
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, cat):
        batch_size = src.size(0)

        # transformer + non-learnable positional encoding
        src = self.encoder(src) * torch.tensor(math.sqrt(self.ninp))
        src = self.pos_encoder(src)

        # embed categorical vars and concat with encoded src
        encoded = self.transformer_encoder(src).view(-1, self.ninp*self.max_len)
        cat_embed = self.cat_embed(cat).view(-1, self.cat_size*self.cat_len)
        features = torch.cat([encoded,cat_embed],dim=1)
        output = self.decoder(features)
        return output
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)