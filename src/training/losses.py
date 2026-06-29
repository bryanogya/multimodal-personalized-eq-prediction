import torch
import torch.nn as nn

def spectral_convergence(pred_curve, target_curve, eps=1e-8):
    sc = (torch.norm(pred_curve - target_curve, p=2, dim=1) / 
        (torch.norm(target_curve, p=2, dim=1) + eps))

    return torch.mean(sc)

def log_spectral_distance(pred_curve, target_curve, eps=1e-8):
    lsd = torch.sqrt(torch.mean((pred_curve - target_curve) ** 2, dim=1) + 
                     eps)

    return torch.mean(lsd)

class SpectralLoss(nn.Module):
    def __init__(self, sc_weight=1.0, lsd_weight=1.0, eps=1e-8):
        super().__init__()
        
        self.sc_weight = sc_weight
        self.lsd_weight = lsd_weight
        self.eps = eps

    def forward(self, pred_curve, target_curve):
        sc = spectral_convergence(
            pred_curve,
            target_curve,
            eps=self.eps
        )

        lsd = log_spectral_distance(
            pred_curve,
            target_curve,
            eps=self.eps
        )

        loss = (self.sc_weight * sc + self.lsd_weight * lsd)

        return loss
    
    def components(self, pred_curve, target_curve):
        sc = spectral_convergence(
            pred_curve,
            target_curve,
            eps=self.eps
        )

        lsd = log_spectral_distance(
            pred_curve,
            target_curve,
            eps=self.eps
        )

        return sc, lsd
    
if __name__ == "__main__":
    spectral_criterion = SpectralLoss()

    pred_curve = torch.randn(4, 128)
    target_curve = torch.randn(4, 128)

    spec_loss = spectral_criterion(pred_curve, target_curve)
    sc, lsd = spectral_criterion.components(pred_curve, target_curve)

    print("=== SPECTRAL LOSS TEST ===")
    print(f"Loss: {spec_loss.item():.6f}")