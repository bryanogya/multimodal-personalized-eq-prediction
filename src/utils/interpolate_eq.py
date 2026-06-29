import torch
import torch.nn as nn

from configs.eq import EQ_FREQS
from configs.frequencies import TARGET_FREQ


class LogFrequencyInterpolator(nn.Module):
    def __init__(self):
        super().__init__()

        log_eq = torch.log10(
            torch.tensor(EQ_FREQS, dtype=torch.float32)
        )

        log_target = torch.log10(
            torch.tensor(TARGET_FREQ, dtype=torch.float32)
        )

        idx_right = torch.searchsorted(
            log_eq,
            log_target
        )

        idx_right = torch.clamp(
            idx_right,
            1,
            len(log_eq) - 1
        )

        idx_left = idx_right - 1

        x0 = log_eq[idx_left]
        x1 = log_eq[idx_right]

        x_target = torch.clamp(
            log_target,
            min=log_eq[0],
            max=log_eq[-1]
        )

        weights = (x_target - x0) / (x1 - x0)

        self.register_buffer("idx_left", idx_left)
        self.register_buffer("idx_right", idx_right)
        self.register_buffer("weights", weights)

    def forward(self, pred_eq):
        """
        pred_eq: (batch, 6)
        return  : (batch, 128)
        """

        y0 = pred_eq[:, self.idx_left]
        y1 = pred_eq[:, self.idx_right]

        pred_curve = y0 + self.weights * (y1 - y0)

        return pred_curve
    
if __name__ == "__main__":
    interpolator = LogFrequencyInterpolator()

    pred_eq = torch.randn(4, 6)

    pred_curve = interpolator(pred_eq)

    print(pred_eq.shape)
    print(pred_curve.shape)