import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralLoss(nn.Module):
    """
    Composite Spectral Loss for Audio Dereverberation.
    Updated to prevent exploitation of clamping and properly handle invalid outputs.
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
        # Only clamp targets (they're always valid from your data)
        target_spec = torch.clamp(target_spec, min=1e-4, max=1.0)
        
        # Check for invalid predictions BEFORE any processing
        invalid_mask = (pred_spec < 0) | (pred_spec > 1)
        invalid_ratio = invalid_mask.float().mean()
        
        # Penalty for invalid outputs (crucial to prevent exploitation)
        validity_penalty = 0
        if self.penalize_invalid and invalid_ratio > 0:
            # Strong penalty that scales with how far outside [0,1] the values are
            negative_penalty = F.relu(-pred_spec).mean() * 2.0  # Penalty for negative
            overflow_penalty = F.relu(pred_spec - 1).mean() * 2.0  # Penalty for >1
            validity_penalty = negative_penalty + overflow_penalty
            
            # Log warning during training
            if invalid_ratio > 0.1:  # More than 10% invalid
                print(f"⚠️ {invalid_ratio:.1%} invalid outputs! Range: [{pred_spec.min():.3f}, {pred_spec.max():.3f}]")
        
        # For stable computation, replace invalid values but track the penalty
        pred_spec_stable = torch.where(
            pred_spec > 0,
            torch.clamp(pred_spec, max=1.5),  # Allow slight overflow but cap it
            torch.ones_like(pred_spec) * 1e-4  # Replace negatives with small positive
        )
        
        # 1. L1 Loss (using stable version)
        l1_loss = F.l1_loss(pred_spec_stable, target_spec)
        
        # 2. MSE Loss (using stable version)
        mse_loss = F.mse_loss(pred_spec_stable, target_spec)
        
        # 3. Log Magnitude Loss (will naturally penalize negative values)
        eps = 1e-4
        # This will be high if pred_spec has negative values (log of small number)
        pred_log = torch.log10(pred_spec_stable + eps)
        target_log = torch.log10(target_spec + eps)
        log_loss = F.smooth_l1_loss(pred_log, target_log, beta=0.01)
        
        # 4. Spectral Convergence
        sc_num = torch.norm(target_spec - pred_spec_stable, p='fro', dim=(-2, -1))
        sc_den = torch.norm(target_spec, p='fro', dim=(-2, -1))
        sc_loss = (sc_num / (sc_den + eps)).mean()
        
        # 5. Time Gradient Loss
        grad_loss_t = 0
        if pred_spec_stable.shape[-1] > 1:
            pred_grad_t = pred_spec_stable[..., 1:] - pred_spec_stable[..., :-1]
            target_grad_t = target_spec[..., 1:] - target_spec[..., :-1]
            grad_loss_t = F.l1_loss(pred_grad_t, target_grad_t)
        
        # 6. Frequency Gradient Loss
        grad_loss_f = 0
        if pred_spec_stable.shape[-2] > 1:
            pred_grad_f = pred_spec_stable[..., 1:, :] - pred_spec_stable[..., :-1, :]
            target_grad_f = target_spec[..., 1:, :] - target_spec[..., :-1, :]
            grad_loss_f = F.l1_loss(pred_grad_f, target_grad_f)
        
        grad_loss = grad_loss_t + grad_loss_f
        
        # 7. Multi-Scale Loss
        multiscale_loss = 0
        scales = [2, 4, 8]
        valid_scales = 0
        
        for scale in scales:
            if pred_spec_stable.shape[-2] >= scale and pred_spec_stable.shape[-1] >= scale:
                pred_down = F.avg_pool2d(pred_spec_stable, kernel_size=scale, stride=scale)
                target_down = F.avg_pool2d(target_spec, kernel_size=scale, stride=scale)
                
                scale_l1 = F.l1_loss(pred_down, target_down)
                
                pred_down_log = torch.log10(pred_down + eps)
                target_down_log = torch.log10(target_down + eps)
                scale_log = F.smooth_l1_loss(pred_down_log, target_down_log, beta=0.01)
                
                multiscale_loss += scale_l1 + 0.3 * scale_log
                valid_scales += 1
        
        if valid_scales > 0:
            multiscale_loss /= valid_scales
        
        # 8. Energy Conservation Loss
        pred_energy = torch.sum(pred_spec_stable ** 2, dim=(-2, -1))
        target_energy = torch.sum(target_spec ** 2, dim=(-2, -1))
        energy_loss = F.l1_loss(pred_energy, target_energy)
        
        # 9. High-Frequency Emphasis
        freq_weights = torch.linspace(0.8, 1.2, pred_spec_stable.shape[-2]).to(pred_spec_stable.device)
        freq_weights = torch.sqrt(freq_weights)
        freq_weights = freq_weights.view(1, 1, -1, 1)
        weighted_pred = pred_spec_stable * freq_weights
        weighted_target = target_spec * freq_weights
        hf_loss = F.smooth_l1_loss(weighted_pred, weighted_target, beta=0.01)
        
        # Determine weights
        if self.adaptive_weights:
            log_weight = torch.sigmoid(self.log_weight) * 0.3
            hf_weight = torch.sigmoid(self.hf_weight) * 0.1
        else:
            # With global normalization, we can use higher weights
            log_weight = 0.20  # Safe with global norm
            hf_weight = 0.05   # Safe with global norm
        
        # Combine all losses
        spectral_loss = (
            0.25 * l1_loss +
            0.15 * mse_loss +
            log_weight * log_loss +
            0.10 * sc_loss +
            0.10 * grad_loss +
            0.10 * multiscale_loss +
            0.05 * energy_loss +
            hf_weight * hf_loss
        )
        
        # Add validity penalty to total loss
        total_loss = spectral_loss + validity_penalty
        
        return total_loss


class MSEWithPenalty(nn.Module):
    """
    MSE loss with penalty for invalid outputs.
    Good for comparison and debugging.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, pred_spec, target_spec):
        # Clamp targets only
        target_spec = torch.clamp(target_spec, min=1e-4, max=1.0)
        
        # Basic MSE
        mse_loss = F.mse_loss(pred_spec, target_spec)
        
        # Penalty for invalid outputs
        negative_penalty = F.relu(-pred_spec).mean() * 5.0
        overflow_penalty = F.relu(pred_spec - 1).mean() * 5.0
        
        total_loss = mse_loss + negative_penalty + overflow_penalty
        
        # Warning for debugging
        if (pred_spec < 0).any():
            invalid_pct = (pred_spec < 0).float().mean() * 100
            print(f"MSE Loss - {invalid_pct:.1f}% negative outputs!")
        
        return total_loss


class CurriculumSpectralLoss(nn.Module):
    """
    Spectral Loss with curriculum learning and proper invalid output handling.
    Gradually introduces complexity while always penalizing invalid outputs.
    """
    def __init__(self, max_epochs=100):
        super().__init__()
        self.max_epochs = max_epochs
        self.current_epoch = 0
        
    def set_epoch(self, epoch):
        """Call this at the start of each epoch"""
        self.current_epoch = epoch
        
    def get_weights(self):
        """Calculate component weights based on current epoch"""
        if self.current_epoch < 10:
            # Early training - focus on basic reconstruction
            return {
                'l1': 0.40,
                'mse': 0.30,
                'log': 0.05,
                'sc': 0.10,
                'grad': 0.10,
                'multiscale': 0.05,
                'energy': 0.00,
                'hf': 0.00,
                'validity': 1.0  # Always penalize invalid outputs
            }
        elif self.current_epoch < 30:
            # Mid training - add more components
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
            # Full training - all components
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
        """
        Forward pass with curriculum learning and validity checks.
        """
        w = self.get_weights()
        
        # Only clamp targets
        target_spec = torch.clamp(target_spec, min=1e-4, max=1.0)
        
        # Validity penalty (always active)
        validity_penalty = 0
        if w['validity'] > 0:
            negative_penalty = F.relu(-pred_spec).mean() * 2.0
            overflow_penalty = F.relu(pred_spec - 1).mean() * 2.0
            validity_penalty = w['validity'] * (negative_penalty + overflow_penalty)
            
            # Check ratio for logging
            invalid_ratio = ((pred_spec < 0) | (pred_spec > 1)).float().mean()
            if invalid_ratio > 0.1:
                print(f"⚠️ Epoch {self.current_epoch}: {invalid_ratio:.1%} invalid outputs!")
        
        # Stabilize predictions for loss computation
        pred_spec_stable = torch.where(
            pred_spec > 0,
            torch.clamp(pred_spec, max=1.5),
            torch.ones_like(pred_spec) * 1e-4
        )
        
        eps = 1e-4
        total_loss = validity_penalty  # Start with penalty
        
        # 1. L1 Loss
        if w['l1'] > 0:
            l1_loss = F.l1_loss(pred_spec_stable, target_spec)
            total_loss += w['l1'] * l1_loss
        
        # 2. MSE Loss
        if w['mse'] > 0:
            mse_loss = F.mse_loss(pred_spec_stable, target_spec)
            total_loss += w['mse'] * mse_loss
        
        # 3. Log Loss
        if w['log'] > 0:
            pred_log = torch.log10(pred_spec_stable + eps)
            target_log = torch.log10(target_spec + eps)
            log_loss = F.smooth_l1_loss(pred_log, target_log, beta=0.01)
            total_loss += w['log'] * log_loss
        
        # 4. Spectral Convergence
        if w['sc'] > 0:
            sc_num = torch.norm(target_spec - pred_spec_stable, p='fro', dim=(-2, -1))
            sc_den = torch.norm(target_spec, p='fro', dim=(-2, -1))
            sc_loss = (sc_num / (sc_den + eps)).mean()
            total_loss += w['sc'] * sc_loss
        
        # 5. Gradient Losses (Time and Frequency)
        if w['grad'] > 0:
            grad_loss = 0
            
            # Time gradient
            if pred_spec_stable.shape[-1] > 1:
                pred_grad_t = pred_spec_stable[..., 1:] - pred_spec_stable[..., :-1]
                target_grad_t = target_spec[..., 1:] - target_spec[..., :-1]
                grad_loss += F.l1_loss(pred_grad_t, target_grad_t)
            
            # Frequency gradient
            if pred_spec_stable.shape[-2] > 1:
                pred_grad_f = pred_spec_stable[..., 1:, :] - pred_spec_stable[..., :-1, :]
                target_grad_f = target_spec[..., 1:, :] - target_spec[..., :-1, :]
                grad_loss += F.l1_loss(pred_grad_f, target_grad_f)
            
            total_loss += w['grad'] * grad_loss
        
        # 6. Multi-scale Loss
        if w['multiscale'] > 0:
            multiscale_loss = 0
            valid_scales = 0
            
            for scale in [2, 4, 8]:
                if pred_spec_stable.shape[-2] >= scale and pred_spec_stable.shape[-1] >= scale:
                    pred_down = F.avg_pool2d(pred_spec_stable, kernel_size=scale, stride=scale)
                    target_down = F.avg_pool2d(target_spec, kernel_size=scale, stride=scale)
                    
                    # L1 at this scale
                    scale_l1 = F.l1_loss(pred_down, target_down)
                    
                    # Log loss at this scale
                    pred_down_log = torch.log10(pred_down + eps)
                    target_down_log = torch.log10(target_down + eps)
                    scale_log = F.smooth_l1_loss(pred_down_log, target_down_log, beta=0.01)
                    
                    multiscale_loss += scale_l1 + 0.3 * scale_log
                    valid_scales += 1
            
            if valid_scales > 0:
                multiscale_loss /= valid_scales
                total_loss += w['multiscale'] * multiscale_loss
        
        # 7. Energy Conservation
        if w['energy'] > 0:
            pred_energy = torch.sum(pred_spec_stable ** 2, dim=(-2, -1))
            target_energy = torch.sum(target_spec ** 2, dim=(-2, -1))
            energy_loss = F.l1_loss(pred_energy, target_energy)
            total_loss += w['energy'] * energy_loss
        
        # 8. High-frequency Emphasis
        if w['hf'] > 0:
            freq_weights = torch.linspace(0.9, 1.1, pred_spec_stable.shape[-2]).to(pred_spec_stable.device)
            freq_weights = freq_weights.view(1, 1, -1, 1)
            weighted_pred = pred_spec_stable * freq_weights
            weighted_target = target_spec * freq_weights
            hf_loss = F.smooth_l1_loss(weighted_pred, weighted_target, beta=0.01)
            total_loss += w['hf'] * hf_loss
        
        return total_loss
