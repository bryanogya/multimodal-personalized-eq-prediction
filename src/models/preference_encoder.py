import torch
import torch.nn as nn

from configs.model import PREFERENCE_INPUT_DIM, PREFERENCE_HIDDEN_DIM, PREFERENCE_EMBED_DIM, PREFERENCE_DROPOUT

class PreferenceEncoder(nn.Module):
    def __init__(self):
        super(PreferenceEncoder, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(
                PREFERENCE_INPUT_DIM, 
                PREFERENCE_HIDDEN_DIM
            ),
            nn.ReLU(),
            nn.Dropout(PREFERENCE_DROPOUT),
            
            nn.Linear(
                PREFERENCE_HIDDEN_DIM, 
                PREFERENCE_EMBED_DIM
            )
        )
        
        self.norm = nn.LayerNorm(
            PREFERENCE_EMBED_DIM
        )

    def forward(self, x):
        x = self.network(x)
        x = self.norm(x)
        return x

# Testing Model Preference Encoder
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pref_encoder = PreferenceEncoder().to(device)

    test_pref = torch.randn(4, 3).to(device)
    test_output = pref_encoder(test_pref)

    print("Preference Encoder Test")
    print(f"Input shape      : {test_pref.shape}")
    print(f"Output shape     : {test_output.shape}")
    print(f"Total Parameters : {sum(p.numel() for p in pref_encoder.parameters()):,}")