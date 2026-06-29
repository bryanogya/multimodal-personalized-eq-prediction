import torch
import torch.nn as nn

from configs.model import (
    AUDIO_EMBED_DIM,
    DEVICE_EMBED_DIM,
    PREFERENCE_EMBED_DIM,
    FUSION_HIDDEN_DIM,
    FUSION_DROPOUT,
    NUM_EQ_BANDS,
)

from configs.frequencies import NUM_BANDS

from src.models.audio_encoder import AudioEncoder
from src.models.device_encoder import DeviceEncoder
from src.models.preference_encoder import PreferenceEncoder


class MultimodalEQModel(nn.Module):
    def __init__(
        self,
        use_audio=True,
        use_device=True,
        use_preference=True
    ):
        super().__init__()

        self.use_audio = use_audio
        self.use_device = use_device
        self.use_preference = use_preference

        fusion_dim = 0

        if self.use_audio:
            self.audio_encoder = AudioEncoder()
            fusion_dim += AUDIO_EMBED_DIM

        if self.use_device:
            self.device_encoder = DeviceEncoder()
            fusion_dim += DEVICE_EMBED_DIM

        if self.use_preference:
            self.preference_encoder = PreferenceEncoder()
            fusion_dim += PREFERENCE_EMBED_DIM

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, FUSION_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(FUSION_DROPOUT),

            nn.Linear(FUSION_HIDDEN_DIM, 128),
            nn.ReLU(),
            nn.Dropout(FUSION_DROPOUT),
        )

        self.eq_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(FUSION_DROPOUT),
            nn.Linear(64, NUM_EQ_BANDS)
        )

        self.curve_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(FUSION_DROPOUT),
            nn.Linear(128, NUM_BANDS)
        )

    def forward(
        self,
        audio=None,
        device=None,
        preference=None
    ):
        embeddings = []

        if self.use_audio:
            embeddings.append(
                self.audio_encoder(audio)
            )

        if self.use_device:
            embeddings.append(
                self.device_encoder(device)
            )

        if self.use_preference:
            embeddings.append(
                self.preference_encoder(preference)
            )

        fused = torch.cat(
            embeddings,
            dim=1
        )

        fused_features = self.fusion(fused)

        pred_eq = self.eq_head(fused_features)

        pred_response_curve = self.curve_head(fused_features)

        return {
            "eq": pred_eq,
            "response_curve": pred_response_curve
        }


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = MultimodalEQModel().to(device)

    batch_size = 4

    audio = torch.randn(batch_size, 1, 128, 256).to(device)
    device_fr = torch.randn(batch_size, 128).to(device)
    preference = torch.randn(batch_size, 3).to(device)

    output = model(
        audio=audio,
        device=device_fr,
        preference=preference
    )

    print("=== MODEL TEST ===")
    print("EQ Output Shape          :", output["eq"].shape)
    print("Curve Output Shape       :", output["response_curve"].shape)
    print("Total Params             :", sum(p.numel() for p in model.parameters()))