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
    
    Args:
        magnitude: Magnitude spectrogram
        phase: Phase spectrogram
        n_fft: FFT window size
        hop_length: Hop length for iSTFT
    
    Returns:
        audio: Reconstructed audio waveform
    """
    # Reconstruct complex spectrogram
    real = magnitude * torch.cos(phase)
    imag = magnitude * torch.sin(phase)
    complex_spec = torch.complex(real, imag)
    
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
    resized = F.interpolate(spec, size=target_size, mode='bilinear', align_corners=False)
    
    return resized


def unresize_spectrogram(spec, original_size):
    """
    Resize spectrogram back to original dimensions
    
    Args:
        spec: Spectrogram tensor
        original_size: Original (height, width)
    
    Returns:
        Resized spectrogram
    """
    resized = F.interpolate(spec, size=original_size, mode='bilinear', align_corners=False)
    return resized
