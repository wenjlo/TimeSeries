from assets.CustomTransformer import TransformerDecoderLayer,generate_square_subsequent_mask
import torch


if __name__ == "__main__":
    d_model = 512  # 特徵維度
    nhead = 8  # 注意力頭數
    dim_feedforward = 2048  # 前饋網路隱藏層維度
    dropout = 0.1  # Dropout 機率

    # 實例化 CustomTransformerDecoderLayer
    decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
    print("CustomTransformerDecoderLayer Model:\n", decoder_layer)

    # 模擬輸入數據
    # tgt: 目標序列 (target_sequence_length, batch_size, d_model)
    # memory: 編碼器輸出 (source_sequence_length, batch_size, d_model)
    tgt_len = 10
    src_len = 12
    batch_size = 4

    tgt_input = torch.randn(tgt_len, batch_size, d_model)
    encoder_output_memory = torch.randn(src_len, batch_size, d_model)

    print(f"\nTarget input shape: {tgt_input.shape}")
    print(f"Encoder memory shape: {encoder_output_memory.shape}")

    # 模擬遮罩
    # tgt_mask (因果遮罩): (tgt_len, tgt_len)
    tgt_mask = generate_square_subsequent_mask(tgt_len)

    # tgt_key_padding_mask: (batch_size, tgt_len)
    # 假設 batch 0 和 2 的最後兩個目標 token 是填充
    tgt_key_padding_mask = torch.zeros(batch_size, tgt_len, dtype=torch.bool)
    tgt_key_padding_mask[0, 8:] = True
    tgt_key_padding_mask[2, 7:] = True

    # memory_key_padding_mask: (batch_size, src_len)
    # 假設 batch 1 和 3 的最後三個源 token 是填充
    memory_key_padding_mask = torch.zeros(batch_size, src_len, dtype=torch.bool)
    memory_key_padding_mask[1, 9:] = True
    memory_key_padding_mask[3, 8:] = True

    print(f"Target mask shape: {tgt_mask.shape}")
    print(f"Target key padding mask shape: {tgt_key_padding_mask.shape}")
    print(f"Memory key padding mask shape: {memory_key_padding_mask.shape}")

    # 執行前向傳播
    output = decoder_layer(tgt_input, encoder_output_memory,
                           tgt_mask=tgt_mask,
                           memory_mask=None,  # 編碼器-解碼器注意力通常不需要額外的attn_mask
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask)

    print(f"\nOutput shape: {output.shape}")

    # 驗證輸出形狀是否正確
    assert output.shape == tgt_input.shape
    print("\nOutput shape matches target input shape, as expected for a Decoder Layer.")
