import torch
import torch.nn.functional as F

# Constants for the pipeline - use these everywhere
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = None  # We use linear spectrogram, not mel
DB_MIN = -80.0  # dB floor
DB_MAX = 0.0    # dB ceiling (relative to max)
TARGET_FRAMES = 512  # Pad/crop time dimension to this
TARGET_FREQ = 1040  # n_fft//2 + 1 = 1025, padded to multiple of 16


def audio_to_spectrogram(audio, n_fft=N_FFT, hop_length=HOP_LENGTH, return_db=True):
    """
    Convert audio waveform to spectrogram with proper dB scaling.
    
    Args:
        audio: Audio tensor [channels, samples] or [samples]
        n_fft: FFT window size (default 2048 for better freq resolution)
        hop_length: Hop length for STFT (default 512)
        return_db: If True, return log-magnitude in dB scale
    
    Returns:
        magnitude: Magnitude spectrogram (linear or dB based on return_db)
        phase: Phase spectrogram (always linear)
    """
    # Handle 1D input
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    
    # Take first channel if stereo
    if audio.shape[0] > 1:
        audio = audio[0:1]
    
    window = torch.hann_window(n_fft, device=audio.device)
    
    # STFT
    spec = torch.stft(
        audio.squeeze(0),  # Remove channel dim for stft
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True
    )
    
    magnitude = torch.abs(spec)
    phase = torch.angle(spec)
    
    if return_db:
        # Convert to dB scale
        # Add small epsilon to avoid log(0)
        magnitude = 20 * torch.log10(magnitude + 1e-8)
        # Clamp to reasonable dB range
        magnitude = torch.clamp(magnitude, min=DB_MIN, max=DB_MAX)
    
    return magnitude, phase


def spectrogram_to_audio(magnitude, phase, n_fft=N_FFT, hop_length=HOP_LENGTH, from_db=True):
    """
    Convert magnitude spectrogram + phase back to audio waveform.
    
    Args:
        magnitude: Magnitude spectrogram (dB or linear)
        phase: Phase spectrogram
        n_fft: FFT window size
        hop_length: Hop length for iSTFT
        from_db: If True, magnitude is in dB scale and needs conversion
    
    Returns:
        audio: Reconstructed audio waveform
    """
    # Handle shape mismatches
    if magnitude.shape != phase.shape:
        print(f"  ⚠️ Shape mismatch: mag {magnitude.shape} vs phase {phase.shape}")
        # Interpolate magnitude to match phase (phase has original shape)
        if magnitude.dim() == 2:
            magnitude = magnitude.unsqueeze(0).unsqueeze(0)
            magnitude = F.interpolate(magnitude, size=phase.shape[-2:], mode='bilinear')
            magnitude = magnitude.squeeze(0).squeeze(0)
        elif magnitude.dim() == 3:
            magnitude = magnitude.unsqueeze(0)
            magnitude = F.interpolate(magnitude, size=phase.shape[-2:], mode='bilinear')
            magnitude = magnitude.squeeze(0)
    
    if from_db:
        # Convert from dB back to linear
        magnitude = 10 ** (magnitude / 20)
    
    # Clamp to valid range
    magnitude = torch.clamp(magnitude, min=0.0)
    
    # Reconstruct complex spectrogram
    complex_spec = magnitude * torch.exp(1j * phase)
    
    window = torch.hann_window(n_fft, device=magnitude.device)
    
    audio = torch.istft(
        complex_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window
    )
    
    return audio


def normalize_db_spectrogram(spec_db):
    """
    Normalize dB spectrogram to [0, 1] range.
    
    Args:
        spec_db: Spectrogram in dB scale (expected range [DB_MIN, DB_MAX])
    
    Returns:
        Normalized spectrogram in [0, 1]
    """
    # Map [DB_MIN, DB_MAX] -> [0, 1]
    spec_norm = (spec_db - DB_MIN) / (DB_MAX - DB_MIN)
    spec_norm = torch.clamp(spec_norm, 0, 1)
    return spec_norm


def denormalize_db_spectrogram(spec_norm):
    """
    Convert normalized [0, 1] spectrogram back to dB scale.
    
    Args:
        spec_norm: Normalized spectrogram in [0, 1]
    
    Returns:
        Spectrogram in dB scale [DB_MIN, DB_MAX]
    """
    spec_db = spec_norm * (DB_MAX - DB_MIN) + DB_MIN
    return spec_db


def pad_spectrogram(spec, target_frames=TARGET_FRAMES, target_freq=TARGET_FREQ):
    """
    Pad spectrogram to target size (no information loss).
    
    Args:
        spec: Spectrogram [freq, time] or [batch, channel, freq, time]
        target_frames: Target time dimension
        target_freq: Target frequency dimension (None = keep original)
    
    Returns:
        Padded spectrogram, original_shape for later cropping
    """
    original_shape = spec.shape
    
    # Handle different input dimensions
    if spec.dim() == 2:
        freq, time = spec.shape
        spec = spec.unsqueeze(0).unsqueeze(0)  # [1, 1, F, T]
    elif spec.dim() == 3:
        spec = spec.unsqueeze(0)  # [1, C, F, T]
    
    _, _, freq, time = spec.shape
    
    # Calculate padding needed
    pad_time = max(0, target_frames - time)
    pad_freq = max(0, (target_freq or freq) - freq)
    
    # Pad: (left, right, top, bottom) for last two dims
    # F.pad expects (left, right, top, bottom) = (time_left, time_right, freq_left, freq_right)
    spec_padded = F.pad(spec, (0, pad_time, 0, pad_freq), mode='constant', value=0)
    
    # Crop if larger than target
    if time > target_frames:
        spec_padded = spec_padded[:, :, :, :target_frames]
    if target_freq and freq > target_freq:
        spec_padded = spec_padded[:, :, :target_freq, :]
    
    return spec_padded.squeeze(0).squeeze(0), original_shape


def unpad_spectrogram(spec, original_shape):
    """
    Remove padding to restore original spectrogram size.
    
    Args:
        spec: Padded spectrogram
        original_shape: Original shape before padding
    
    Returns:
        Cropped spectrogram matching original dimensions
    """
    if len(original_shape) == 2:
        orig_freq, orig_time = original_shape
        return spec[:orig_freq, :orig_time]
    elif len(original_shape) == 3:
        _, orig_freq, orig_time = original_shape
        return spec[:, :orig_freq, :orig_time]
    elif len(original_shape) == 4:
        _, _, orig_freq, orig_time = original_shape
        return spec[:, :, :orig_freq, :orig_time]
    return spec


def process_audio_for_model(audio, sr=16000):
    """
    Full pipeline: audio -> model-ready spectrogram.
    
    Args:
        audio: Audio tensor [samples] or [channels, samples]
        sr: Sample rate (for reference, not used in processing)
    
    Returns:
        spec_norm: Normalized spectrogram ready for model [1, F, T]
        phase: Phase for reconstruction
        original_shape: For unpadding later
    """
    # Get dB spectrogram
    spec_db, phase = audio_to_spectrogram(audio, return_db=True)
    
    # Normalize to [0, 1]
    spec_norm = normalize_db_spectrogram(spec_db)
    
    # Pad to fixed size
    spec_padded, original_shape = pad_spectrogram(spec_norm, target_frames=TARGET_FRAMES)
    
    # Add channel dimension [F, T] -> [1, F, T]
    if spec_padded.dim() == 2:
        spec_padded = spec_padded.unsqueeze(0)
    
    return spec_padded, phase, original_shape


def model_output_to_audio(spec_norm, phase, original_shape):
    """
    Full pipeline: model output -> audio.
    
    Args:
        spec_norm: Model output spectrogram [1, F, T] or [F, T], normalized [0, 1]
        phase: Original phase spectrogram
        original_shape: Original spectrogram shape before padding
    
    Returns:
        audio: Reconstructed audio waveform
    """
    # Remove channel dim if present
    if spec_norm.dim() == 3:
        spec_norm = spec_norm.squeeze(0)
    
    # Unpad
    spec_norm = unpad_spectrogram(spec_norm, original_shape)
    
    # Denormalize to dB
    spec_db = denormalize_db_spectrogram(spec_norm)
    
    # Convert to audio
    audio = spectrogram_to_audio(spec_db, phase, from_db=True)
    
    return audio


# For backwards compatibility - but prefer the new functions
def resize_spectrogram(spec, target_size=(256, 256)):
    """DEPRECATED: Use pad_spectrogram instead to avoid information loss."""
    print("⚠️ Warning: resize_spectrogram loses information. Use pad_spectrogram instead.")
    
    if spec.dim() == 2:
        spec = spec.unsqueeze(0).unsqueeze(0)
    elif spec.dim() == 3:
        spec = spec.unsqueeze(0)
    
    resized = F.interpolate(spec, size=target_size, mode='bilinear')
    return resized


def unresize_spectrogram(spec, original_size):
    """DEPRECATED: Use unpad_spectrogram instead."""
    resized = F.interpolate(spec, size=original_size, mode='bilinear')
    return resized
