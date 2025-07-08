import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class Transformer(nn.Module):
    """
    max_len : maximum sequence length
    feature_size : embedding dimension
    """
    def __init__(self, feature_size=250, num_layers=1, dropout=0.1):
        super(Transformer, self).__init__()
        self.model_type = 'Transformer'

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        #self.decoder = nn.Linear(feature_size, 1)
        self.relu = nn.ReLU()
        self.layer_1 = nn.Linear(feature_size,feature_size//2)
        self.layer_2 = nn.Linear(feature_size//2,feature_size//4)
        self.decoder = nn.Linear(feature_size//4,1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.layer_1(output)
        output = self.relu(output)
        output = self.layer_2(output)
        output = self.relu(output)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        """
        # 這對解碼器的自注意力機制至關重要，確保預測t+1時看不到t+2及以後的數據

        :param sz: input sequence length
        :return: mask tensor
        example : input sequence length = 4,

          return  tensor([[0., -inf, -inf, -inf],
                        [0., 0., -inf, -inf],
                        [0., 0., 0., -inf],
                        [0., 0., 0., 0.]])
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

