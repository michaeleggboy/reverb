import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralLoss(nn.Module):
    """
    Hybrid Spectral Loss for Audio Dereverberation.
    Combines best practices from multiple perspectives:
    - Decoupled validity penalty (prevents gradient conflicts)
    - Stable log computation for audio (dB scale)
    - Multi-scale pooling (perceptually important)
    - Cleaner implementation for faster convergence
    """
    def __init__(self, adaptive_weights=False, penalize_invalid=True):
        super().__init__()
        self.adaptive_weights = adaptive_weights
        self.penalize_invalid = penalize_invalid
        
        if adaptive_weights:
            # Learnable weights that can be optimized during training
            self.log_weight = nn.Parameter(torch.tensor(0.1))
            self.hf_weight = nn.Parameter(torch.tensor(0.02))
        
    def forward(self, pred_spec, target_spec):
        """
        Args:
            pred_spec: [B, C, F, T] predicted magnitude spectrogram
            target_spec: [B, C, F, T] target magnitude spectrogram
        """
        # Ensure targets are valid (they should be from your data)
        target_spec = torch.clamp(target_spec, min=1e-4, max=1.0)
        
        # =============================================================
        # DECOUPLED VALIDITY PENALTY (calculate on raw predictions)
        # =============================================================
        validity_penalty = 0
        if self.penalize_invalid:
            # Calculate penalties on RAW predictions (not clamped)
            negative_penalty = F.relu(-pred_spec).mean() * 2.0
            overflow_penalty = F.relu(pred_spec - 1.0).mean() * 2.0
            validity_penalty = negative_penalty + overflow_penalty
            
            # Optional logging for debugging
            invalid_ratio = ((pred_spec < 0) | (pred_spec > 1)).float().mean()
            if invalid_ratio > 0.1:
                print(f"⚠️ {invalid_ratio:.1%} invalid outputs! Range: [{pred_spec.min():.3f}, {pred_spec.max():.3f}]")
        
        # =============================================================
        # STABLE PREDICTIONS for loss computation (not for penalty)
        # =============================================================
        # Clamp predictions to valid range for stable loss computation
        # This doesn't affect gradients from validity penalty
        pred_stable = torch.clamp(pred_spec, min=1e-4, max=1.0)
        
        # =============================================================
        # AUDIO-SPECIFIC LOSSES
        # =============================================================
        eps = 1e-7  # Small epsilon for numerical stability
        
        # 1. L1 Loss (base reconstruction)
        l1_loss = F.l1_loss(pred_stable, target_spec)
        
        # 2. MSE Loss (smooth gradients)
        mse_loss = F.mse_loss(pred_stable, target_spec)
        
        # 3. Stable Log Magnitude Loss (perceptually important for audio)
        # Extra safety: ensure no zeros before log
        pred_log_input = torch.clamp(pred_stable, min=eps)
        target_log_input = torch.clamp(target_spec, min=eps)
        pred_log = torch.log10(pred_log_input + eps)
        target_log = torch.log10(target_log_input + eps)
        log_loss = F.smooth_l1_loss(pred_log, target_log, beta=0.01)
        
        # 4. Spectral Convergence (scale-invariant)
        sc_num = torch.norm(target_spec - pred_stable, p='fro', dim=(-2, -1))
        sc_den = torch.norm(target_spec, p='fro', dim=(-2, -1)) + eps
        sc_loss = (sc_num / sc_den).mean()
        
        # 5. Improved Gradient Loss (vectorized for efficiency)
        grad_loss = 0
        # Time gradient
        if pred_stable.shape[-1] > 1:
            pred_grad_t = torch.diff(pred_stable, dim=-1)
            target_grad_t = torch.diff(target_spec, dim=-1)
            grad_loss += F.l1_loss(pred_grad_t, target_grad_t)
        
        # Frequency gradient
        if pred_stable.shape[-2] > 1:
            pred_grad_f = torch.diff(pred_stable, dim=-2)
            target_grad_f = torch.diff(target_spec, dim=-2)
            grad_loss += F.l1_loss(pred_grad_f, target_grad_f)
        
        # 6. Multi-Scale Loss (perceptually important for audio)
        multiscale_loss = 0
        scales = [2, 4, 8]
        valid_scales = 0
        
        for scale in scales:
            if pred_stable.shape[-2] >= scale and pred_stable.shape[-1] >= scale:
                # Pooling for multi-resolution analysis
                pred_down = F.avg_pool2d(pred_stable, kernel_size=scale, stride=scale)
                target_down = F.avg_pool2d(target_spec, kernel_size=scale, stride=scale)
                
                # L1 at this scale
                scale_l1 = F.l1_loss(pred_down, target_down)
                
                # Log loss at this scale (with safety)
                pred_down_safe = torch.clamp(pred_down, min=eps)
                target_down_safe = torch.clamp(target_down, min=eps)
                pred_down_log = torch.log10(pred_down_safe + eps)
                target_down_log = torch.log10(target_down_safe + eps)
                scale_log = F.smooth_l1_loss(pred_down_log, target_down_log, beta=0.01)
                
                multiscale_loss += scale_l1 + 0.3 * scale_log
                valid_scales += 1
        
        if valid_scales > 0:
            multiscale_loss /= valid_scales
        
        # 7. Energy Conservation Loss
        pred_energy = torch.sum(pred_stable ** 2, dim=(-2, -1))
        target_energy = torch.sum(target_spec ** 2, dim=(-2, -1))
        energy_loss = F.l1_loss(pred_energy, target_energy)
        
        # 8. High-Frequency Emphasis (cleaner implementation)
        # Linear frequency weighting with sqrt for gentler emphasis
        freq_weights = torch.linspace(0.9, 1.1, pred_stable.shape[-2], device=pred_stable.device)
        freq_weights = torch.sqrt(freq_weights).view(1, 1, -1, 1)
        weighted_pred = pred_stable * freq_weights
        weighted_target = target_spec * freq_weights
        hf_loss = F.smooth_l1_loss(weighted_pred, weighted_target, beta=0.01)
        
        # =============================================================
        # WEIGHT DETERMINATION
        # =============================================================
        if self.adaptive_weights:
            # Learnable weights bounded by sigmoid
            log_weight = torch.sigmoid(self.log_weight) * 0.3  # Max 0.3
            hf_weight = torch.sigmoid(self.hf_weight) * 0.1   # Max 0.1
        else:
            # Fixed weights optimized for global normalization
            log_weight = 0.20
            hf_weight = 0.05
        
        # =============================================================
        # COMBINE LOSSES (decoupled approach)
        # =============================================================
        # Audio reconstruction losses (computed on stable predictions)
        spectral_loss = (
            0.25 * l1_loss +          # Base reconstruction
            0.15 * mse_loss +         # Smooth gradients
            log_weight * log_loss +   # Perceptual (dB scale)
            0.10 * sc_loss +          # Scale invariance
            0.10 * grad_loss +        # Structure preservation
            0.10 * multiscale_loss +  # Multi-resolution
            0.05 * energy_loss +      # Energy conservation
            hf_weight * hf_loss       # High-frequency clarity
        )
        
        # Total loss = audio losses + validity penalty (decoupled)
        total_loss = spectral_loss + validity_penalty
        
        return total_loss


class MSEWithPenalty(nn.Module):
    """
    Simple MSE loss with validity penalty for comparison.
    Uses same decoupled approach.
    """
    def __init__(self, penalize_invalid=True):
        super().__init__()
        self.penalize_invalid = penalize_invalid
        
    def forward(self, pred_spec, target_spec):
        target_spec = torch.clamp(target_spec, min=1e-4, max=1.0)
        
        # Decoupled validity penalty
        validity_penalty = 0
        if self.penalize_invalid:
            negative_penalty = F.relu(-pred_spec).mean() * 5.0
            overflow_penalty = F.relu(pred_spec - 1.0).mean() * 5.0
            validity_penalty = negative_penalty + overflow_penalty
        
        # MSE on clamped predictions
        pred_stable = torch.clamp(pred_spec, min=1e-4, max=1.0)
        mse_loss = F.mse_loss(pred_stable, target_spec)
        
        return mse_loss + validity_penalty


class CurriculumSpectralLoss(nn.Module):
    """
    Curriculum version with hybrid improvements.
    Gradually introduces complex components while maintaining stability.
    """
    def __init__(self, max_epochs=100):
        super().__init__()
        self.max_epochs = max_epochs
        self.current_epoch = 0
        
    def set_epoch(self, epoch):
        """Update epoch for curriculum scheduling"""
        self.current_epoch = epoch
        
    def get_weights(self):
        """Progressive weight scheduling"""
        if self.current_epoch < 10:
            # Early: Focus on basic reconstruction
            return {
                'l1': 0.40,
                'mse': 0.30,
                'log': 0.05,
                'sc': 0.10,
                'grad': 0.10,
                'multiscale': 0.05,
                'energy': 0.00,
                'hf': 0.00,
                'validity': 1.0
            }
        elif self.current_epoch < 30:
            # Mid: Add perceptual components
            return {
                'l1': 0.30,
                'mse': 0.20,
                'log': 0.10,
                'sc': 0.10,
                'grad': 0.10,
                'multiscale': 0.10,
                'energy': 0.05,
                'hf': 0.05,
                'validity': 1.0
            }
        else:
            # Late: Full complexity
            return {
                'l1': 0.25,
                'mse': 0.15,
                'log': 0.15,
                'sc': 0.10,
                'grad': 0.10,
                'multiscale': 0.10,
                'energy': 0.05,
                'hf': 0.10,
                'validity': 1.0
            }
    
    def forward(self, pred_spec, target_spec):
        """Forward with curriculum and hybrid improvements"""
        w = self.get_weights()
        target_spec = torch.clamp(target_spec, min=1e-4, max=1.0)
        
        # Decoupled validity penalty (always active)
        validity_penalty = 0
        if w['validity'] > 0:
            negative_penalty = F.relu(-pred_spec).mean() * 2.0
            overflow_penalty = F.relu(pred_spec - 1.0).mean() * 2.0
            validity_penalty = w['validity'] * (negative_penalty + overflow_penalty)
        
        # Stable predictions for loss computation
        pred_stable = torch.clamp(pred_spec, min=1e-4, max=1.0)
        
        eps = 1e-7
        total_loss = validity_penalty  # Start with penalty
        
        # Add weighted components based on curriculum
        if w['l1'] > 0:
            total_loss += w['l1'] * F.l1_loss(pred_stable, target_spec)
        
        if w['mse'] > 0:
            total_loss += w['mse'] * F.mse_loss(pred_stable, target_spec)
        
        if w['log'] > 0:
            pred_log = torch.log10(torch.clamp(pred_stable, min=eps) + eps)
            target_log = torch.log10(torch.clamp(target_spec, min=eps) + eps)
            total_loss += w['log'] * F.smooth_l1_loss(pred_log, target_log, beta=0.01)
        
        if w['sc'] > 0:
            sc_num = torch.norm(target_spec - pred_stable, p='fro', dim=(-2, -1))
            sc_den = torch.norm(target_spec, p='fro', dim=(-2, -1)) + eps
            total_loss += w['sc'] * (sc_num / sc_den).mean()
        
        if w['grad'] > 0:
            grad_loss = 0
            if pred_stable.shape[-1] > 1:
                grad_loss += F.l1_loss(torch.diff(pred_stable, dim=-1), 
                                       torch.diff(target_spec, dim=-1))
            if pred_stable.shape[-2] > 1:
                grad_loss += F.l1_loss(torch.diff(pred_stable, dim=-2), 
                                       torch.diff(target_spec, dim=-2))
            total_loss += w['grad'] * grad_loss
        
        if w['multiscale'] > 0:
            multiscale_loss = 0
            valid_scales = 0
            for scale in [2, 4, 8]:
                if pred_stable.shape[-2] >= scale and pred_stable.shape[-1] >= scale:
                    pred_down = F.avg_pool2d(pred_stable, kernel_size=scale, stride=scale)
                    target_down = F.avg_pool2d(target_spec, kernel_size=scale, stride=scale)
                    multiscale_loss += F.l1_loss(pred_down, target_down)
                    valid_scales += 1
            if valid_scales > 0:
                total_loss += w['multiscale'] * (multiscale_loss / valid_scales)
        
        if w['energy'] > 0:
            pred_energy = torch.sum(pred_stable ** 2, dim=(-2, -1))
            target_energy = torch.sum(target_spec ** 2, dim=(-2, -1))
            total_loss += w['energy'] * F.l1_loss(pred_energy, target_energy)
        
        if w['hf'] > 0:
            freq_weights = torch.sqrt(torch.linspace(0.9, 1.1, pred_stable.shape[-2], 
                                                    device=pred_stable.device))
            freq_weights = freq_weights.view(1, 1, -1, 1)
            total_loss += w['hf'] * F.smooth_l1_loss(pred_stable * freq_weights, 
                                                     target_spec * freq_weights, beta=0.01)
        
        return total_loss
