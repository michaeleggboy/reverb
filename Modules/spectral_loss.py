import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralLoss(nn.Module):
    """
    Composite Spectral Loss for Audio Dereverberation.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, pred_spec, target_spec):
        """
        Args:
            pred_spec: [B, C, F, T] predicted magnitude spectrogram
            target_spec: [B, C, F, T] target magnitude spectrogram
        """
        # 1. L1 Loss (more robust to outliers than MSE)
        l1_loss = F.l1_loss(pred_spec, target_spec)
        
        # 2. MSE Loss (for stability and smooth gradients)
        mse_loss = F.mse_loss(pred_spec, target_spec)
        
        # 3. Log Magnitude Loss (crucial for low-energy reverb tails)
        eps = 1e-7
        pred_log = torch.log(pred_spec.clamp(min=eps))
        target_log = torch.log(target_spec.clamp(min=eps))
        log_loss = F.l1_loss(pred_log, target_log)
        
        # 4. Spectral Convergence (scale-invariant loss)
        sc_num = torch.norm(target_spec - pred_spec, p='fro', dim=(-2, -1))
        sc_den = torch.norm(target_spec, p='fro', dim=(-2, -1))
        sc_loss = (sc_num / (sc_den + eps)).mean()
        
        # 5. Time Gradient Loss (preserves temporal structure)
        grad_loss_t = 0
        if pred_spec.shape[-1] > 1:
            pred_grad_t = pred_spec[..., 1:] - pred_spec[..., :-1]
            target_grad_t = target_spec[..., 1:] - target_spec[..., :-1]
            grad_loss_t = F.l1_loss(pred_grad_t, target_grad_t)
        
        # 6. Frequency Gradient Loss (preserves harmonic structure)
        grad_loss_f = 0
        if pred_spec.shape[-2] > 1:
            pred_grad_f = pred_spec[..., 1:, :] - pred_spec[..., :-1, :]
            target_grad_f = target_spec[..., 1:, :] - target_spec[..., :-1, :]
            grad_loss_f = F.l1_loss(pred_grad_f, target_grad_f)
        
        grad_loss = grad_loss_t + grad_loss_f
        
        # 7. Multi-Scale Loss in Spectrogram Domain
        multiscale_loss = 0
        scales = [2, 4, 8]
        valid_scales = 0
        
        for scale in scales:
            if pred_spec.shape[-2] >= scale and pred_spec.shape[-1] >= scale:
                # Downsample with average pooling
                pred_down = F.avg_pool2d(pred_spec, kernel_size=scale, stride=scale)
                target_down = F.avg_pool2d(target_spec, kernel_size=scale, stride=scale)
                
                # L1 loss at this scale
                scale_l1 = F.l1_loss(pred_down, target_down)
                
                # Log loss at this scale (important for reverb)
                pred_down_log = torch.log(pred_down.clamp(min=eps))
                target_down_log = torch.log(target_down.clamp(min=eps))
                scale_log = F.l1_loss(pred_down_log, target_down_log)
                
                multiscale_loss += scale_l1 + 0.3 * scale_log
                valid_scales += 1
        
        if valid_scales > 0:
            multiscale_loss /= valid_scales
        
        # 8. Energy Conservation Loss (total energy should be similar)
        pred_energy = torch.sum(pred_spec ** 2, dim=(-2, -1))
        target_energy = torch.sum(target_spec ** 2, dim=(-2, -1))
        energy_loss = F.l1_loss(pred_energy, target_energy)
        
        # 9. High-Frequency Emphasis (important for clarity)
        # Weight higher frequencies more (they're perceptually important)
        freq_weights = torch.linspace(0.5, 1.5, pred_spec.shape[-2]).to(pred_spec.device)
        freq_weights = freq_weights.view(1, 1, -1, 1)
        weighted_pred = pred_spec * freq_weights
        weighted_target = target_spec * freq_weights
        hf_loss = F.l1_loss(weighted_pred, weighted_target)
        
        # Combine all losses with optimized weights for dereverberation
        total_loss = (
            0.25 * l1_loss +          # Base L1
            0.15 * mse_loss +         # MSE for stability
            0.20 * log_loss +         # Log magnitude (important!)
            0.10 * sc_loss +          # Spectral convergence
            0.10 * grad_loss +        # Gradients (time + freq)
            0.10 * multiscale_loss +  # Multi-scale features
            0.05 * energy_loss +      # Energy conservation
            0.05 * hf_loss            # High-frequency emphasis
        )
        
        return total_loss
