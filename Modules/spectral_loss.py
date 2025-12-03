import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss for spectrograms"""
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16]
        self.vgg = vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False
        
        # VGG normalization stats
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, pred, target):
        # Repeat single channel to 3 channels
        pred_3ch = pred.repeat(1, 3, 1, 1)
        target_3ch = target.repeat(1, 3, 1, 1)
        
        # Normalize for VGG
        pred_norm = (pred_3ch - self.mean) / self.std
        target_norm = (target_3ch - self.mean) / self.std
        
        # Get features
        pred_feat = self.vgg(pred_norm)
        target_feat = self.vgg(target_norm)
        
        return F.l1_loss(pred_feat, target_feat)


class SpectralLoss(nn.Module):
    """
    Hybrid Spectral Loss for Audio Dereverberation.
    Combines best practices from multiple perspectives:
    - Decoupled validity penalty (prevents gradient conflicts)
    - Stable log computation for audio (dB scale)
    - Multi-scale pooling (perceptually important)
    - VGG perceptual loss
    """
    def __init__(self, adaptive_weights=False, penalize_invalid=True, use_perceptual=True):
        super().__init__()
        self.adaptive_weights = adaptive_weights
        self.penalize_invalid = penalize_invalid
        self.use_perceptual = use_perceptual
        
        self.hf_weight_min = 0.10
        self.log_weight_min = 0.15
        
        if use_perceptual:
            self.perceptual = PerceptualLoss()
        
    def forward(self, pred_spec, target_spec):
        """
        Args:
            pred_spec: [B, C, F, T] predicted magnitude spectrogram
            target_spec: [B, C, F, T] target magnitude spectrogram
        """
        target_spec = torch.clamp(target_spec, min=1e-4, max=1.0)
        
        # =============================================================
        # DECOUPLED VALIDITY PENALTY (calculate on raw predictions)
        # =============================================================
        validity_penalty = 0
        if self.penalize_invalid:
            negative_penalty = F.relu(-pred_spec).mean() * 2.0
            overflow_penalty = F.relu(pred_spec - 1.0).mean() * 2.0
            validity_penalty = negative_penalty + overflow_penalty
            
            invalid_ratio = ((pred_spec < 0) | (pred_spec > 1)).float().mean()
            if invalid_ratio > 0.1:
                print(f"⚠️ {invalid_ratio:.1%} invalid outputs! Range: [{pred_spec.min():.3f}, {pred_spec.max():.3f}]")
        
        # =============================================================
        # STABLE PREDICTIONS for loss computation
        # =============================================================
        pred_stable = torch.clamp(pred_spec, min=1e-4, max=1.0)
        
        # =============================================================
        # AUDIO-SPECIFIC LOSSES
        # =============================================================
        eps = 1e-7
        
        # 1. L1 Loss
        l1_loss = F.l1_loss(pred_stable, target_spec)
        
        # 2. MSE Loss
        mse_loss = F.mse_loss(pred_stable, target_spec)
        
        # 3. Log Magnitude Loss
        pred_log = torch.log10(pred_stable + eps)
        target_log = torch.log10(target_spec + eps)
        log_loss = F.smooth_l1_loss(pred_log, target_log, beta=0.01)
        
        # 4. Spectral Convergence
        sc_num = torch.norm(target_spec - pred_stable, p='fro', dim=(-2, -1))
        sc_den = torch.norm(target_spec, p='fro', dim=(-2, -1)) + eps
        sc_loss = (sc_num / sc_den).mean()
        
        # 5. Gradient Loss
        grad_loss = 0
        if pred_stable.shape[-1] > 1:
            pred_grad_t = torch.diff(pred_stable, dim=-1)
            target_grad_t = torch.diff(target_spec, dim=-1)
            grad_loss += F.l1_loss(pred_grad_t, target_grad_t)
        if pred_stable.shape[-2] > 1:
            pred_grad_f = torch.diff(pred_stable, dim=-2)
            target_grad_f = torch.diff(target_spec, dim=-2)
            grad_loss += F.l1_loss(pred_grad_f, target_grad_f)
        
        # 6. Multi-Scale Loss
        multiscale_loss = 0
        scales = [2, 4, 8]
        valid_scales = 0
        for scale in scales:
            if pred_stable.shape[-2] >= scale and pred_stable.shape[-1] >= scale:
                pred_down = F.avg_pool2d(pred_stable, kernel_size=scale, stride=scale)
                target_down = F.avg_pool2d(target_spec, kernel_size=scale, stride=scale)
                scale_l1 = F.l1_loss(pred_down, target_down)
                pred_down_log = torch.log10(pred_down + eps)
                target_down_log = torch.log10(target_down + eps)
                scale_log = F.smooth_l1_loss(pred_down_log, target_down_log, beta=0.01)
                multiscale_loss += scale_l1 + 0.3 * scale_log
                valid_scales += 1
        if valid_scales > 0:
            multiscale_loss /= valid_scales
        
        # 7. Energy Conservation Loss
        pred_energy = torch.mean(pred_stable ** 2, dim=(-2, -1))
        target_energy = torch.mean(target_spec ** 2, dim=(-2, -1))
        energy_loss = F.l1_loss(pred_energy, target_energy)
        
        # 8. High-Frequency Emphasis
        freq_weights = torch.linspace(0.9, 1.1, pred_stable.shape[-2], device=pred_stable.device)
        freq_weights = torch.sqrt(freq_weights).view(1, 1, -1, 1)
        weighted_pred = pred_stable * freq_weights
        weighted_target = target_spec * freq_weights
        hf_loss = F.smooth_l1_loss(weighted_pred, weighted_target, beta=0.01)
        
        # 9. Perceptual Loss (VGG)
        perceptual_loss = 0
        if self.use_perceptual:
            perceptual_loss = self.perceptual(pred_stable, target_spec)
        
        # =============================================================
        # WEIGHT DETERMINATION
        # =============================================================
        if self.adaptive_weights:
            # Compute weights based on ERROR - higher error = higher weight
            with torch.no_grad():
                hf_error = F.l1_loss(weighted_pred, weighted_target).item()
                log_error = F.l1_loss(pred_log, target_log).item()
                l1_error = l1_loss.item()
                
                # Normalize errors relative to l1 (baseline)
                total_error = hf_error + log_error + l1_error + 1e-7
                
                # Higher error → higher weight (inverse of what you had)
                hf_weight = max(self.hf_weight_min, 0.3 * (hf_error / total_error))
                log_weight = max(self.log_weight_min, 0.4 * (log_error / total_error))
        else:
            log_weight = 0.25
            hf_weight = 0.15
        
        # =============================================================
        # COMBINE LOSSES
        # =============================================================
        spectral_loss = (
            0.20 * l1_loss +
            0.10 * mse_loss +
            log_weight * log_loss +
            0.10 * sc_loss +
            0.10 * grad_loss +
            0.10 * multiscale_loss +
            0.05 * energy_loss +
            hf_weight * hf_loss +
            0.10 * perceptual_loss
        )
        
        total_loss = spectral_loss + validity_penalty
        
        return total_loss


class MSEWithPenalty(nn.Module):
    """Simple MSE loss with validity penalty for comparison."""
    def __init__(self, penalize_invalid=True):
        super().__init__()
        self.penalize_invalid = penalize_invalid
        
    def forward(self, pred_spec, target_spec):
        target_spec = torch.clamp(target_spec, min=1e-4, max=1.0)
        
        validity_penalty = 0
        if self.penalize_invalid:
            negative_penalty = F.relu(-pred_spec).mean() * 5.0
            overflow_penalty = F.relu(pred_spec - 1.0).mean() * 5.0
            validity_penalty = negative_penalty + overflow_penalty
        
        pred_stable = torch.clamp(pred_spec, min=1e-4, max=1.0)
        mse_loss = F.mse_loss(pred_stable, target_spec)
        
        return mse_loss + validity_penalty
