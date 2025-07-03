from idlelib import query

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttention(nn.Module):
    def __init__(self,embed_dim,num_heads,dropout=0.1):
        """
        Args:
            embed_dim: 輸入和輸出的特徵維度 (d_model)。
            num_heads: 多頭注意力機制中的頭數。
            dropout: 注意力權重上的 Dropout 機率。
        """
        super(MultiheadAttention,self).__init__()
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
        src_len,_, = key.shape
        # 1. 線性投影並切分多頭
        # q, k, v shape after proj: (L, N, embed_dim)
        # Reshape to (L, N, num_heads, head_dim) and permute to (num_heads, N, L, head_dim)
        # 再轉換成每個頭獨立的 Q, K, V (num_heads * N, L, head_dim)
        # PyTorch MultiheadAttention 內部會將 N*num_heads 合併為一個維度處理
        # 為了簡化，我們這裡保持 (num_heads, N, seq_len, head_dim)

        q = self.q_proj(query).view(tgt_len, batch_size * self.num_heads, self.head_dim).transpose(0,1)  # (N*num_heads, L, head_dim)
        k = self.k_proj(key).view(src_len, batch_size * self.num_heads, self.head_dim).transpose(0,1)  # (N*num_heads, S, head_dim)
        v = self.v_proj(value).view(src_len, batch_size * self.num_heads, self.head_dim).transpose(0,1)  # (N*num_heads, S, head_dim)

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
