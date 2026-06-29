import torch
import torch.nn as nn

from src.models.audio_encoder import AudioEncoder
from src.models.device_encoder import DeviceEncoder
from src.models.preference_encoder import PreferenceEncoder

from configs.model import (
    AUDIO_EMBED_DIM,
    DEVICE_EMBED_DIM,
    PREFERENCE_EMBED_DIM,
    FUSION_DROPOUT,
    NUM_EQ_BANDS
)


class MultimodalEQModel(nn.Module):
    """
    Multimodal neural network for EQ band prediction.

    The model can combine audio features, device frequency response,
    and user preference embeddings, then predict EQ gain values for
    each frequency band.
    """
    def __init__(
        self,
        use_audio=True,
        use_device=True,
        use_preference=True
    ):
        """
        Initialize the multimodal EQ prediction model.

        Args:
            use_audio: Whether to use the audio encoder branch.
            use_device: Whether to use the device response encoder branch.
            use_preference: Whether to use the preference encoder branch.
        """
        super().__init__()

        self.use_audio = use_audio
        self.use_device = use_device
        self.use_preference = use_preference

        # Encoders
        if self.use_audio:
            self.audio_encoder = AudioEncoder()

        if self.use_device:
            self.device_encoder = DeviceEncoder()

        if self.use_preference:
            self.preference_encoder = PreferenceEncoder()

        # Dynamic Fusion Dimension
        fusion_dim = 0

        if self.use_audio:
            fusion_dim += AUDIO_EMBED_DIM

        if self.use_device:
            fusion_dim += DEVICE_EMBED_DIM

        if self.use_preference:
            fusion_dim += PREFERENCE_EMBED_DIM

        # Fusion + Regression Head
        self.fusion_head = nn.Sequential(

            nn.Linear(
                fusion_dim,
                256
            ),
            nn.ReLU(),
            nn.Dropout(FUSION_DROPOUT),

            nn.Linear(
                256,
                128
            ),
            nn.ReLU(),
            nn.Dropout(FUSION_DROPOUT),

            nn.Linear(
                128,
                64
            ),
            nn.ReLU(),
            nn.Dropout(FUSION_DROPOUT),

            nn.Linear(
                64,
                NUM_EQ_BANDS
            )
        )

    def forward(
        self,
        audio=None,
        device=None,
        preference=None
    ):
        """
        Run a forward pass through the selected modality branches.

        Args:
            audio: Audio input tensor for the audio encoder.
            device: Device response tensor for the device encoder.
            preference: Preference vector tensor for the preference encoder.

        Returns:
            Predicted EQ gain values for each EQ frequency band.

        Raises:
            ValueError: If a required modality input is missing.
        """
        embeddings = []

        if self.use_audio:
            if audio is None:
                raise ValueError(
                    "Audio input is required."
                )

            audio_embedding = self.audio_encoder(audio)
            embeddings.append(audio_embedding)

        if self.use_device:
            if device is None:
                raise ValueError(
                    "Device input is required."
                )

            device_embedding = self.device_encoder(device)
            embeddings.append(device_embedding)

        if self.use_preference:
            if preference is None:
                raise ValueError(
                    "Preference input is required."
                )

            preference_embedding = self.preference_encoder(preference)
            embeddings.append(preference_embedding)

        fused_embedding = torch.cat(embeddings,dim=1)
        eq_prediction = self.fusion_head(fused_embedding)

        return eq_prediction
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalEQModel().to(device)

    batch_size = 4

    audio = torch.randn(
        batch_size,
        1,
        128,
        259
    ).to(device)

    device_fr = torch.randn(
        batch_size,
        128
    ).to(device)

    preference = torch.randn(
        batch_size,
        3
    ).to(device)

    output = model(
        audio=audio,
        device=device_fr,
        preference=preference
    )
    
    def count_params(model):
        """
        Count the number of trainable parameters in a model.

        Args:
            model: PyTorch model.

        Returns:
            Number of trainable parameters.
        """
        return sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

    print("=== MODEL TEST ===")
    print(f"Output Shape : {output.shape}")
    print(f"Audio Encoder      : {count_params(model.audio_encoder):,}")
    print(f"Device Encoder     : {count_params(model.device_encoder):,}")
    print(f"Preference Encoder : {count_params(model.preference_encoder):,}")
    print(f"Total Model        : {count_params(model):,}")