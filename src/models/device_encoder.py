import torch
import torch.nn as nn

from configs.model import DEVICE_INPUT_DIM, DEVICE_EMBED_DIM, DEVICE_HIDDEN_DIM, DEVICE_DROPOUT

class DeviceEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc_layers = nn.Sequential(
            nn.Linear(
                DEVICE_INPUT_DIM,
                DEVICE_HIDDEN_DIM
            ),
            nn.ReLU(),
            nn.Dropout(DEVICE_DROPOUT),

            nn.Linear(
                DEVICE_HIDDEN_DIM,
                DEVICE_EMBED_DIM
            )
        )

        self.norm = nn.LayerNorm(
            DEVICE_EMBED_DIM
        )

    def forward(self, x):
        x = self.fc_layers(x)
        x = self.norm(x)
        return x 

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    device_encoder = DeviceEncoder(input_dim=128, embed_dim=128).to(device)

    test_device = torch.randn(4, 128).to(device)
    test_output = device_encoder(test_device)

    print("\nUji Coba Device Encoder (MLP)")
    print(f"Input shape      : {test_device.shape}")
    print(f"Output shape     : {test_output.shape}")
    print(f"Total Parameters : {sum(p.numel() for p in device_encoder.parameters()):,}")