import torch
import torch.nn as nn

from configs.model import NUM_BANDS, AUDIO_EMBED_DIM, AUDIO_NUM_HEADS, AUDIO_NUM_LAYERS, AUDIO_PATCH_SIZE, AUDIO_DROPOUT 


class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        
        # 1. Fix input time dimension agar sequence length konsisten
        self.fix_time = nn.AdaptiveAvgPool2d((NUM_BANDS, 256))
        
        # 2. Convolutional Patch Embedding 
        self.patch_conv = nn.Conv2d(
            in_channels=1, 
            out_channels=AUDIO_EMBED_DIM, 
            kernel_size=AUDIO_PATCH_SIZE, 
            stride=AUDIO_PATCH_SIZE
        )
        
        # 3. Positional Encoding (Learnable)
        self.seq_len = (NUM_BANDS // AUDIO_PATCH_SIZE) * (AUDIO_EMBED_DIM // AUDIO_PATCH_SIZE)
        self.pos_encoding = nn.Parameter(torch.randn(1, self.seq_len, AUDIO_EMBED_DIM))
        
        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=AUDIO_EMBED_DIM, nhead=AUDIO_NUM_HEADS, 
            dim_feedforward=AUDIO_EMBED_DIM * AUDIO_NUM_LAYERS, dropout=AUDIO_DROPOUT,
            batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=AUDIO_NUM_LAYERS)
        
        # 5. Output Projection
        self.output_projection = nn.Linear(AUDIO_EMBED_DIM, AUDIO_EMBED_DIM)
        self.dropout = nn.Dropout(AUDIO_DROPOUT)
        self.norm = nn.LayerNorm(AUDIO_EMBED_DIM)

    def forward(self, x):
        # x: (batch, 1, n_mels, time)
        
        # 1. Fix time dimension → (batch, 1, 128, 256)
        x = self.fix_time(x)
        
        # 2. Patch Embedding → (batch, embed_dim, 8, 16)
        x = self.patch_conv(x)
        
        # 3. Flatten & transpose ke (batch, seq_len, embed_dim) → (batch, 128, 256)
        x = x.flatten(2).transpose(1, 2)
        
        # 4. Tambah positional encoding
        x = x + self.pos_encoding
        
        # 5. Transformer
        x = self.transformer(x)
        
        # 6. Global pooling + proyeksi akhir
        x = x.mean(dim=1)               # (batch, 256)
        x = self.norm(x)
        x = self.output_projection(x)
        x = self.dropout(x)
        return x

# ==========================================
# Testing Model Cabang Audio
# ==========================================
if __name__ == "__main__":
    # Gunakan GPU jika tersedia
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio_encoder = AudioEncoder().to(device)

    # Simulasi data Mel-spectrogram: 4 batch, 1 channel, 128 n_mels, durasi 300 time frames
    test_audio = torch.randn(4, 1, 128, 300).to(device)

    test_output = audio_encoder(test_audio)

    print("=== Audio Encoder Test Berhasil ===")
    print(f"Input shape          : {test_audio.shape}")
    print(f"Output shape         : {test_output.shape}")
    print(f"Total Parameters     : {sum(p.numel() for p in audio_encoder.parameters()):,}")