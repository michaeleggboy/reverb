import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralLoss(nn.Module):
    """
    Composite Spectral Loss for Audio Dereverberation.
    Updated with improved stability for log magnitude and HF components.
    """
    def __init__(self, adaptive_weights=False):
        super().__init__()
        self.adaptive_weights = adaptive_weights
        
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
        # Ensure numerical stability throughout
        pred_spec = torch.clamp(pred_spec, min=1e-4, max=1.0)
        target_spec = torch.clamp(target_spec, min=1e-4, max=1.0)
        
        # 1. L1 Loss (more robust to outliers than MSE)
        l1_loss = F.l1_loss(pred_spec, target_spec)
        
        # 2. MSE Loss (for stability and smooth gradients)
        mse_loss = F.mse_loss(pred_spec, target_spec)
        
        # 3. IMPROVED Log Magnitude Loss (more stable)
        eps = 1e-4
        # Use log10 instead of natural log (gentler gradients)
        pred_log = torch.log10(pred_spec + eps)
        target_log = torch.log10(target_spec + eps)
        # Use SmoothL1 instead of L1 (less sensitive to outliers)
        log_loss = F.smooth_l1_loss(pred_log, target_log, beta=0.01)
        
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
                
                # More stable log loss at this scale
                pred_down_log = torch.log10(pred_down + eps)
                target_down_log = torch.log10(target_down + eps)
                scale_log = F.smooth_l1_loss(pred_down_log, target_down_log, beta=0.01)
                
                multiscale_loss += scale_l1 + 0.3 * scale_log
                valid_scales += 1
        
        if valid_scales > 0:
            multiscale_loss /= valid_scales
        
        # 8. Energy Conservation Loss (total energy should be similar)
        pred_energy = torch.sum(pred_spec ** 2, dim=(-2, -1))
        target_energy = torch.sum(target_spec ** 2, dim=(-2, -1))
        energy_loss = F.l1_loss(pred_energy, target_energy)
        
        # 9. IMPROVED High-Frequency Emphasis (gentler weighting)
        freq_weights = torch.linspace(0.8, 1.2, pred_spec.shape[-2]).to(pred_spec.device)
        # Apply sqrt for gentler increase
        freq_weights = torch.sqrt(freq_weights)
        freq_weights = freq_weights.view(1, 1, -1, 1)
        weighted_pred = pred_spec * freq_weights
        weighted_target = target_spec * freq_weights
        # Use SmoothL1 for HF loss too
        hf_loss = F.smooth_l1_loss(weighted_pred, weighted_target, beta=0.01)
        
        # Determine weights
        if self.adaptive_weights:
            # Learnable weights with sigmoid to bound them
            log_weight = torch.sigmoid(self.log_weight) * 0.3  # Max 0.3
            hf_weight = torch.sigmoid(self.hf_weight) * 0.1   # Max 0.1
        else:
            # Fixed weights - reduced for problematic components
            log_weight = 0.10  # Reduced from 0.20
            hf_weight = 0.02   # Reduced from 0.05
        
        # Combine all losses with improved weights
        total_loss = (
            0.30 * l1_loss +          # Increased base L1
            0.20 * mse_loss +         # Increased MSE for stability
            log_weight * log_loss +   # Reduced/adaptive log weight
            0.10 * sc_loss +          # Spectral convergence
            0.10 * grad_loss +        # Gradients (time + freq)
            0.15 * multiscale_loss +  # Increased multi-scale
            0.05 * energy_loss +      # Energy conservation
            hf_weight * hf_loss       # Reduced/adaptive HF weight
        )
        
        return total_loss


class CurriculumSpectralLoss(nn.Module):
    """
    Spectral Loss with curriculum learning - gradually increases difficulty.
    Start with easier objectives, progressively add harder components.
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
        progress = min(self.current_epoch / self.max_epochs, 1.0)
        
        # Start with basic losses, gradually add complex ones
        if self.current_epoch < 10:
            # Early training - focus on basic reconstruction
            return {
                'l1': 0.40,
                'mse': 0.30,
                'log': 0.05,  # Very low initially
                'sc': 0.10,
                'grad': 0.10,
                'multiscale': 0.05,
                'energy': 0.00,
                'hf': 0.00    # No HF emphasis initially
            }
        elif self.current_epoch < 30:
            # Mid training - add more components
            return {
                'l1': 0.30,
                'mse': 0.20,
                'log': 0.10,  # Gradually increase
                'sc': 0.10,
                'grad': 0.10,
                'multiscale': 0.10,
                'energy': 0.05,
                'hf': 0.05    # Start adding HF
            }
        else:
            # Full training - all components
            return {
                'l1': 0.25,
                'mse': 0.15,
                'log': 0.15,  # Still conservative
                'sc': 0.10,
                'grad': 0.10,
                'multiscale': 0.10,
                'energy': 0.05,
                'hf': 0.10    # Full HF weight
            }
    
    def forward(self, pred_spec, target_spec):
        """
        Args:
            pred_spec: [B, C, F, T] predicted magnitude spectrogram
            target_spec: [B, C, F, T] target magnitude spectrogram
        """
        # Get current weights
        w = self.get_weights()
        
        # Ensure numerical stability
        pred_spec = torch.clamp(pred_spec, min=1e-4, max=1.0)
        target_spec = torch.clamp(target_spec, min=1e-4, max=1.0)
        
        eps = 1e-4
        total_loss = 0
        
        # Basic losses (always active)
        if w['l1'] > 0:
            l1_loss = F.l1_loss(pred_spec, target_spec)
            total_loss += w['l1'] * l1_loss
            
        if w['mse'] > 0:
            mse_loss = F.mse_loss(pred_spec, target_spec)
            total_loss += w['mse'] * mse_loss
        
        # Log loss (gradually introduced)
        if w['log'] > 0:
            pred_log = torch.log10(pred_spec + eps)
            target_log = torch.log10(target_spec + eps)
            log_loss = F.smooth_l1_loss(pred_log, target_log, beta=0.01)
            total_loss += w['log'] * log_loss
        
        # Spectral convergence
        if w['sc'] > 0:
            sc_num = torch.norm(target_spec - pred_spec, p='fro', dim=(-2, -1))
            sc_den = torch.norm(target_spec, p='fro', dim=(-2, -1))
            sc_loss = (sc_num / (sc_den + eps)).mean()
            total_loss += w['sc'] * sc_loss
        
        # Gradient losses
        if w['grad'] > 0:
            grad_loss = 0
            if pred_spec.shape[-1] > 1:
                pred_grad_t = pred_spec[..., 1:] - pred_spec[..., :-1]
                target_grad_t = target_spec[..., 1:] - target_spec[..., :-1]
                grad_loss += F.l1_loss(pred_grad_t, target_grad_t)
            if pred_spec.shape[-2] > 1:
                pred_grad_f = pred_spec[..., 1:, :] - pred_spec[..., :-1, :]
                target_grad_f = target_spec[..., 1:, :] - target_spec[..., :-1, :]
                grad_loss += F.l1_loss(pred_grad_f, target_grad_f)
            total_loss += w['grad'] * grad_loss
        
        # Multi-scale loss
        if w['multiscale'] > 0:
            multiscale_loss = 0
            valid_scales = 0
            for scale in [2, 4, 8]:
                if pred_spec.shape[-2] >= scale and pred_spec.shape[-1] >= scale:
                    pred_down = F.avg_pool2d(pred_spec, kernel_size=scale, stride=scale)
                    target_down = F.avg_pool2d(target_spec, kernel_size=scale, stride=scale)
                    multiscale_loss += F.l1_loss(pred_down, target_down)
                    valid_scales += 1
            if valid_scales > 0:
                total_loss += w['multiscale'] * (multiscale_loss / valid_scales)
        
        # Energy conservation (later stages)
        if w['energy'] > 0:
            pred_energy = torch.sum(pred_spec ** 2, dim=(-2, -1))
            target_energy = torch.sum(target_spec ** 2, dim=(-2, -1))
            energy_loss = F.l1_loss(pred_energy, target_energy)
            total_loss += w['energy'] * energy_loss
        
        # High-frequency emphasis (later stages)
        if w['hf'] > 0:
            freq_weights = torch.linspace(0.9, 1.1, pred_spec.shape[-2]).to(pred_spec.device)
            freq_weights = freq_weights.view(1, 1, -1, 1)
            weighted_pred = pred_spec * freq_weights
            weighted_target = target_spec * freq_weights
            hf_loss = F.smooth_l1_loss(weighted_pred, weighted_target, beta=0.01)
            total_loss += w['hf'] * hf_loss
        
        return total_loss
