# --- 測試範例 ---
from assets.CustomTransformer import TransformerEncoderLayer
import torch

if __name__ == "__main__":
    d_model = 512       # 特徵維度
    nhead = 8           # 注意力頭數
    dim_feedforward = 2048 # 前饋網路隱藏層維度
    dropout = 0.1       # Dropout 機率

    # 實例化 CustomTransformerEncoderLayer
    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
    print("CustomTransformerEncoderLayer Model:\n", encoder_layer)

    # 模擬輸入數據
    # (sequence_length, batch_size, d_model)
    seq_len = 10
    batch_size = 4
    input_data = torch.randn(seq_len, batch_size, d_model)
    print(f"\nInput data shape: {input_data.shape}")

    # 模擬注意力遮罩 (例如，如果不需要遮罩，可以為 None 或全為 False/0)
    # src_mask: (seq_len, seq_len), boolean or float('-inf')
    src_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1) # 簡單的上三角遮罩範例
    #src_mask = None # 編碼器通常不需要因果遮罩

    # 模擬填充遮罩 (True 表示該位置為填充，應被忽略)
    # src_key_padding_mask: (batch_size, seq_len)
    # 假設 batch 中的第一個和第三個序列的最後兩個 token 是填充
    src_key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    src_key_padding_mask[0, 8:] = True
    src_key_padding_mask[2, 7:] = True
    print(f"Key padding mask:\n{src_key_padding_mask}")


    # 執行前向傳播
    output = encoder_layer(input_data, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

    print(f"\nOutput shape: {output.shape}")

    # 驗證輸出形狀是否正確
    assert output.shape == input_data.shape
    print("\nOutput shape matches input shape, as expected for an Encoder Layer.")
