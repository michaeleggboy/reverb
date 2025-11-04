import torch
import torch.nn.functional as F


def audio_to_spectrogram(audio, n_fft=512, hop_length=256):
    """
    Convert audio waveform to spectrogram
    
    Args:
        audio: Audio tensor [channels, samples]
        n_fft: FFT window size
        hop_length: Hop length for STFT
    
    Returns:
        magnitude: Magnitude spectrogram
        phase: Phase spectrogram
    """
    # Create Hann window
    window = torch.hann_window(n_fft)
    
    # Compute STFT
    spec = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True
    )
    
    # Separate magnitude and phase
    magnitude = torch.abs(spec)
    phase = torch.angle(spec)
    
    return magnitude, phase


def spectrogram_to_audio(magnitude, phase, n_fft=512, hop_length=256):
    """
    Convert magnitude spectrogram + phase back to audio waveform
    HANDLES SHAPE MISMATCHES AUTOMATICALLY
    
    Args:
        magnitude: Magnitude spectrogram
        phase: Phase spectrogram
        n_fft: FFT window size
        hop_length: Hop length for iSTFT
    
    Returns:
        audio: Reconstructed audio waveform
    """
    
    # ===== CRITICAL: Ensure exact shape match =====
    if magnitude.shape != phase.shape:
        print("  ⚠️ Shape mismatch detected - fixing...")
        print(f"     Magnitude: {magnitude.shape}")
        print(f"     Phase: {phase.shape}")
        
        # Force magnitude to match phase dimensions exactly
        target_shape = phase.shape
        
        # Add batch dim if needed for interpolation
        needs_batch = magnitude.dim() == 3
        if needs_batch:
            magnitude = magnitude.unsqueeze(0)
        
        # Resize to match phase
        magnitude = F.interpolate(
            magnitude,
            size=target_shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        
        # Remove batch dim if we added it
        if needs_batch:
            magnitude = magnitude.squeeze(0)
        
        print(f"     Fixed to: {magnitude.shape}")
    
    # Final verification
    assert magnitude.shape == phase.shape, \
        f"Shape mismatch after fix: magnitude {magnitude.shape} vs phase {phase.shape}"
    
    # Check for invalid values
    if torch.isnan(magnitude).any() or torch.isinf(magnitude).any():
        print("  ⚠️ NaN/Inf in magnitude - replacing with zeros")
        magnitude = torch.nan_to_num(magnitude, nan=0.0, posinf=1.0, neginf=0.0)
    
    if torch.isnan(phase).any() or torch.isinf(phase).any():
        print("  ⚠️ NaN/Inf in phase - replacing with zeros")
        phase = torch.nan_to_num(phase, nan=0.0, posinf=3.14, neginf=-3.14)
    
    # Ensure magnitude is non-negative
    magnitude = torch.clamp(magnitude, min=0.0)
    
    # Reconstruct complex spectrogram using polar form
    complex_spec = magnitude * torch.exp(1j * phase)
    
    # Create Hann window
    window = torch.hann_window(n_fft)
    
    # Inverse STFT
    audio = torch.istft(
        complex_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window
    )
    
    return audio


def resize_spectrogram(spec, target_size=(256, 256)):
    """
    Resize spectrogram to target size
    
    Args:
        spec: Spectrogram tensor
        target_size: Target (height, width)
    
    Returns:
        Resized spectrogram
    """
    # Ensure correct dimensions [batch, channels, height, width]
    if spec.dim() == 2:
        spec = spec.unsqueeze(0).unsqueeze(0)  # [H, W] -> [1, 1, H, W]
    elif spec.dim() == 3:
        spec = spec.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
    
    # Resize
    resized = F.interpolate(
        spec, 
        size=target_size, 
        mode='bilinear', 
        align_corners=False
    )
    
    return resized


def unresize_spectrogram(spec, original_size):
    """
    Resize spectrogram back to original dimensions
    
    Args:
        spec: Spectrogram tensor [batch, channels, H, W]
        original_size: Original (height, width) tuple
    
    Returns:
        Resized spectrogram
    """
    resized = F.interpolate(
        spec, 
        size=original_size, 
        mode='bilinear', 
        align_corners=False
    )
    
    return resized