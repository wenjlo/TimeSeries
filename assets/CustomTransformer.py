from idlelib import query

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from assets.utils import PositionalEncoding

class MultiheadAttention(nn.Module):
    def __init__(self,embed_dim,num_heads,dropout=0.1):
        """
        Args:
            embed_dim: 輸入和輸出的特徵維度 (d_model)。
            num_heads: 多頭注意力機制中的頭數。
            dropout: 注意力權重上的 Dropout 機率。
        """
        super(MultiheadAttention,self).__init__()
        self.batch_first = False
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p=dropout)
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim 必須能被 num_heads 整除"

        """
        Q,K,V [input dim =  embed_dim, ouput dim = embed_dim]
        """
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # final output
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.query_proj.weight)
        self.query_proj.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.key_proj.weight)
        self.key_proj.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.value_proj.weight)
        self.value_proj.bias.data.fill_(0)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        Args:
            query: 查詢張量 (L, N, E)
            key: 鍵張量 (S, N, E)
            value: 值張量 (S, N, E)
                其中：
                L = query sequence length (查詢序列長度)
                S = key/value sequence length (鍵/值序列長度)
                N = batch size (批次大小)
                E = embed_dim (嵌入維度)

            attn_mask: 注意力遮罩 (L, S) 或 (N*num_heads, L, S)。用於遮蔽某些注意力權重。
                       通常用於解碼器的因果遮罩。
                       如果 `attn_mask` 是一個 float 類型，被遮蔽的位置應為 `float('-inf')`。
                       如果 `attn_mask` 是一個 bool 類型，被遮蔽的位置應為 `True`。
            key_padding_mask: 鍵填充遮罩 (N, S)。用於指示鍵序列中的填充 (padding) token。
                              True 表示該位置是填充，應被忽略。
        Returns:
            output: 多頭注意力計算後的輸出 (L, N, E)
            attn_weights: 所有頭的平均注意力權重 (N, L, S) (可選，通常用於分析)
        """

        tgt_len,batch_size,embed_dim = query.shape
        src_len, _, _ = key.shape
        # 1. 線性投影並切分多頭
        # q, k, v shape after proj: (L, N, embed_dim)
        # Reshape to (L, N, num_heads, head_dim) and permute to (num_heads, N, L, head_dim)
        # 再轉換成每個頭獨立的 Q, K, V (num_heads * N, L, head_dim)
        # PyTorch MultiheadAttention 內部會將 N*num_heads 合併為一個維度處理
        # 為了簡化，我們這裡保持 (num_heads, N, seq_len, head_dim)

        q = self.query_proj(query).view(tgt_len, batch_size * self.num_heads, self.head_dim).transpose(0,1)  # (N*num_heads, L, head_dim)
        k = self.key_proj(key).view(src_len, batch_size * self.num_heads, self.head_dim).transpose(0,1)  # (N*num_heads, S, head_dim)
        v = self.value_proj(value).view(src_len, batch_size * self.num_heads, self.head_dim).transpose(0,1)  # (N*num_heads, S, head_dim)

        # 2. 計算縮放點積注意力
        # (N*num_heads, L, S) = (N*num_heads, L, head_dim) X (N*num_heads, head_dim, S)
        attn_scores = torch.matmul(q, k.transpose(-2, -1))

        # 縮放 避免 softmax 輸出不要太極端,造成梯度消失 (Paper Vaswani et al., 2017)
        attn_scores = attn_scores / (self.head_dim ** 0.5)
        """
        應用注意力遮罩 (attn_mask)
        example: tensor([[0., -inf, -inf, -inf],
                         [0., 0., -inf, -inf],
                          [0., 0., 0., -inf],
                           [0., 0., 0., 0.]])
        attn_mask 通常是 (L, S) 或 (N*num_heads, L, S)
        """
        if attn_mask is not None:
            if attn_mask.ndim == 2: # 如果是 (L, S) 的通用遮罩
                attn_scores = attn_scores + attn_mask.unsqueeze(0) # Unsqueeze(0) for num_heads * batch_size
            else: # 如果是 (N*num_heads, L, S) 已經是多頭形狀的遮罩
                attn_scores = attn_scores + attn_mask

        # 應用鍵填充遮罩 (key_padding_mask)
        # key_padding_mask: (N, S) -> 轉換為 (N*num_heads, 1, S)
        # True 的位置在 attn_scores 中會被設為 -inf
        if key_padding_mask is not None:
            # Expand key_padding_mask to (N*num_heads, 1, S)
            expanded_key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1).expand(-1, self.num_heads, -1, -1).reshape(batch_size * self.num_heads, 1, src_len)
            attn_scores = attn_scores.masked_fill(expanded_key_padding_mask, float('-inf'))

        # Softmax 得到注意力權重(You can choose what ever act fun anything you want)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (N*num_heads, L, S)
        attn_weights = self.dropout(attn_weights)
        # 乘以 Value
        # attn_output: (N*num_heads, L, head_dim)
        attn_output = torch.matmul(attn_weights, v)
        # 3. 拼接所有頭的輸出並進行最終線性投影
        # 將多頭的輸出合併回 (L, N, embed_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, batch_size, embed_dim)

        # 最終線性投影
        output = self.out_proj(attn_output)
        # 注意力權重可以返回，通常用於視覺化或分析
        # 我們將所有頭的平均注意力權重返回 (N, L, S)
        avg_attn_weights = attn_weights.view(self.num_heads, batch_size, tgt_len, src_len).mean(dim=0)
        return output, avg_attn_weights



class TransformerEncoderLayer(nn.Module):
    def __init__(self,d_model,n_heads,dim_feedforward=1024,dropout=0.1,activation="relu"):
        """
        Args:
            d_model: 輸入和輸出的特徵維度 (embedding dimension)。
            nhead: 多頭注意力機制中的頭數。
            dim_feedforward: 前饋網路中隱藏層的維度。
            dropout: Dropout 的機率。
            activation: 前饋網路中的激活函數 ('relu' 或 'gelu')。
        """
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward,d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.gelu()
        else:
            raise RuntimeError(f"沒有這個 {activation} 激勵函數")

    def forward(self, src,src_mask=None,src_key_padding_mask=None):
        """
        Args:
            src: 輸入序列張量，形狀為 (sequence_length, batch_size, d_model)。
            src_mask: 注意力遮罩，形狀為 (sequence_length, sequence_length)。用於遮蔽某些位置的注意力。
                      例如，在解碼器中用於因果遮罩 (causal mask)。
            src_key_padding_mask: 鍵填充遮罩，形狀為 (batch_size, sequence_length)。
                                  用於指示哪些 token 是填充 (padding) 並應被忽略。
                                  True 表示要遮蔽/忽略該位置。
        """
        # --- 1. 自注意力子層 ---
        # 多頭自注意力：query, key, value 都來自 src
        # src_key_padding_mask 應用於 key (即 src)
        # output of self_attn: (sequence_length, batch_size, d_model)
        attn_output, _ = self.self_attn(src, src, src,
                                        attn_mask=src_mask,
                                        key_padding_mask=src_key_padding_mask)

        # 殘差連接 + Dropout + 層歸一化
        # src + attn_output: 殘差連接
        # self.dropout1(attn_output): 應用 dropout
        # self.norm1(...): 層歸一化
        src = src + self.dropout1(attn_output)  # (residual network)
        src = self.norm1(src)  # (layer normalization)

        # --- 2. 前饋神經網路 (FFN) 子層 ---
        # 線性層 1 -> 激活函數 -> Dropout -> 線性層 2
        ffn_output = self.linear2(self.dropout(self.activation(self.linear1(src))))

        # 殘差連接 + Dropout + 層歸一化
        src = src + self.dropout2(ffn_output)  # (residual connection)
        src = self.norm2(src)  # (layer normalization)

        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        """
        Args:
            d_model: 輸入和輸出的特徵維度 (embedding dimension)。
            nhead: 多頭注意力機制中的頭數。
            dim_feedforward: 前饋網路中隱藏層的維度。
            dropout: Dropout 的機率。
            activation: 前饋網路中的激活函數 ('relu' 或 'gelu')。
        """
        super(TransformerDecoderLayer, self).__init__()

        # --- 1. 遮蔽多頭自注意力子層 ---
        # query, key, value 都來自 target sequence (tgt)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        # --- 2. 編碼器-解碼器多頭注意力子層 (Cross-Attention) ---
        # query 來自 target sequence 的上一步輸出，key, value 來自 encoder 的記憶體 (memory)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        # --- 3. 前饋神經網路 (FFN) 子層 ---
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout_ffn = nn.Dropout(dropout) # FFN 內部的 dropout
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # --- 4. 層歸一化 (Layer Normalization) ---
        self.norm1 = nn.LayerNorm(d_model) # masked self-attn output norm
        self.norm2 = nn.LayerNorm(d_model) # cross-attn output norm
        self.norm3 = nn.LayerNorm(d_model) # FFN output norm

        # --- 5. Dropout 層 (用於殘差連接後) ---
        self.dropout1 = nn.Dropout(dropout) # after masked self-attn
        self.dropout2 = nn.Dropout(dropout) # after cross-attn
        self.dropout3 = nn.Dropout(dropout) # after FFN

        # --- 6. 激活函數 ---
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise RuntimeError(f"activation should be relu/gelu, not {activation}")

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Args:
            tgt: 目標序列張量，形狀為 (target_sequence_length, batch_size, d_model)。
            memory: 編碼器輸出張量，形狀為 (source_sequence_length, batch_size, d_model)。
            tgt_mask: 目標序列的注意力遮罩 (因果遮罩)，形狀為 (tgt_len, tgt_len)。True 表示遮蔽。
            memory_mask: 編碼器-解碼器注意力遮罩，形狀為 (tgt_len, src_len)。通常為 None。
            tgt_key_padding_mask: 目標序列的鍵填充遮罩，形狀為 (batch_size, tgt_len)。True 表示填充。
            memory_key_padding_mask: 記憶體（編碼器輸出）的鍵填充遮罩，形狀為 (batch_size, src_len)。True 表示填充。
        """

        # --- 1. 遮蔽自注意力子層 ---
        # query, key, value 都來自 tgt
        # tgt_mask: 因果遮罩 (上三角遮罩)，防止偷看未來 token
        # tgt_key_padding_mask: 填充遮罩，忽略填充 token
        masked_attn_output, _ = self.self_attn(tgt, tgt, tgt,
                                               attn_mask=tgt_mask,
                                               key_padding_mask=tgt_key_padding_mask)

        # 殘差連接 + Dropout + 層歸一化
        tgt = tgt + self.dropout1(masked_attn_output)
        tgt = self.norm1(tgt)
        # 此時的 tgt 已經是經過第一層自注意力和歸一化的結果，將作為 Cross-Attention 的 Query

        # --- 2. 編碼器-解碼器注意力子層 (Cross-Attention) ---
        # query 來自 tgt (上一步的輸出)，key 和 value 來自 memory (Encoder 的輸出)
        # memory_mask: 可選的記憶體注意力遮罩
        # memory_key_padding_mask: 關鍵！用於遮蔽 encoder 輸出中的填充 token
        cross_attn_output, _ = self.multihead_attn(tgt, memory, memory,
                                                   attn_mask=memory_mask,
                                                   key_padding_mask=memory_key_padding_mask)

        # 殘差連接 + Dropout + 層歸一化
        tgt = tgt + self.dropout2(cross_attn_output)
        tgt = self.norm2(tgt)
        # 此時的 tgt 已經是經過 Cross-Attention 和歸一化的結果

        # --- 3. 前饋神經網路 (FFN) 子層 ---
        ffn_output = self.linear2(self.dropout_ffn(self.activation(self.linear1(tgt))))

        # 殘差連接 + Dropout + 層歸一化
        tgt = tgt + self.dropout3(ffn_output)
        tgt = self.norm3(tgt)

        return tgt



class Transformer(nn.Module):
    def __init__(self,src_vocab_size, tgt_vocab_size ,d_model,n_heads):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoderLayer(d_model,n_heads,dim_feedforward=1024,dropout=0.1,activation="relu")
        self.decoder = TransformerDecoderLayer(d_model, n_heads, dim_feedforward=1024, dropout=0.1, activation="relu")
        self.pos_encoding = PositionalEncoding(d_model, src_vocab_size)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def create_padding_mask(self, seq, pad_idx):
        mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
        return mask  # Shape: (batch_size, 1, 1, seq_len)

    def create_look_ahead_mask(self, seq_len, device):
        look_ahead_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(device)
        return ~look_ahead_mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, seq_len)
    def forward(self, src, trg, src_pad_idx, trg_pad_idx):
        # Create masks
        src_mask = self.create_padding_mask(src, src_pad_idx)  # For encoder self-attention & decoder cross-attention
        trg_padding_mask = self.create_padding_mask(trg, trg_pad_idx)  # For decoder self-attention
        trg_look_ahead_mask = self.create_look_ahead_mask(trg.size(1), trg.device).unsqueeze(0)  # For decoder self-attention
        print(trg_padding_mask.shape)
        print(trg_look_ahead_mask.shape)
        # Combine decoder self-attention masks:
        # We need to mask padding AND future tokens
        trg_mask = trg_padding_mask & trg_look_ahead_mask

        # Encoder forward pass
        enc_output = self.encoder(src, src_mask)

        # Decoder forward pass
        dec_output = self.decoder(trg, enc_output, trg_mask, src_mask)

        # Final linear layer
        output = self.fc_out(dec_output)  # (batch_size, trg_seq_len, trg_vocab_size)
        return output
