import torch
import torch.nn.functional as F

# Constants for the pipeline
N_FFT = 2048
HOP_LENGTH = 512
TARGET_FRAMES = 512
TARGET_FREQ = 1040  # n_fft//2 + 1 = 1025, padded to multiple of 16
DB_MIN = -60
DB_MAX = 0.0


def audio_to_spectrogram(audio, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """
    Convert audio waveform to dB spectrogram with reference scaling.
    
    Args:
        audio: Audio tensor [channels, samples] or [samples]
        n_fft: FFT window size (default 2048)
        hop_length: Hop length for STFT (default 512)
    
    Returns:
        magnitude_db: dB-scaled magnitude spectrogram [DB_MIN, 0]
        phase: Phase spectrogram
        ref: Reference magnitude for reconstruction
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
        audio.squeeze(0),
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True
    )
    
    magnitude = torch.abs(spec)
    phase = torch.angle(spec)
    
    # Store reference (max magnitude) for reconstruction
    ref = magnitude.max()
    
    # Convert to dB relative to reference
    magnitude_db = 20 * torch.log10(magnitude / ref + 1e-8)
    
    # Clamp to [DB_MIN, 0] dB range
    magnitude_db = torch.clamp(magnitude_db, min=DB_MIN, max=DB_MAX)
    
    return magnitude_db, phase, ref


def spectrogram_to_audio(magnitude_db, phase, ref, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """
    Convert dB spectrogram + phase back to audio waveform.
    
    Args:
        magnitude_db: dB-scaled magnitude spectrogram [DB_MIN, 0]
        phase: Phase spectrogram
        ref: Reference magnitude from audio_to_spectrogram
        n_fft: FFT window size
        hop_length: Hop length for iSTFT
    
    Returns:
        audio: Reconstructed audio waveform
    """
    # Handle shape mismatches by cropping
    if magnitude_db.shape != phase.shape:
        min_freq = min(magnitude_db.shape[-2], phase.shape[-2])
        min_time = min(magnitude_db.shape[-1], phase.shape[-1])
        magnitude_db = magnitude_db[..., :min_freq, :min_time]
        phase = phase[..., :min_freq, :min_time]
    
    # Convert from dB back to linear using reference
    magnitude = ref * (10 ** (magnitude_db / 20))
    
    # Clamp to valid range
    magnitude = torch.clamp(magnitude, min=0.0)
    
    # Reconstruct complex spectrogram
    complex_spec = magnitude * torch.exp(1j * phase)
    
    window = torch.hann_window(n_fft, device=magnitude_db.device)
    
    audio = torch.istft(
        complex_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window
    )
    
    return audio


def normalize_db_spectrogram(spec_db):
    """
    Normalize dB spectrogram from [DB_MIN, 0] to [0, 1].
    
    Args:
        spec_db: Spectrogram in dB scale [DB_MIN, 0]
    
    Returns:
        Normalized spectrogram in [0, 1]
    """
    # Map [DB_MIN, 0] -> [0, 1]
    spec_norm = (spec_db - DB_MIN) / (DB_MAX - DB_MIN)
    spec_norm = torch.clamp(spec_norm, 0, 1)
    return spec_norm


def denormalize_db_spectrogram(spec_norm):
    """
    Convert normalized [0, 1] spectrogram back to dB scale.
    
    Args:
        spec_norm: Normalized spectrogram in [0, 1]
    
    Returns:
        Spectrogram in dB scale [DB_MIN, 0]
    """
    spec_db = spec_norm * (DB_MAX - DB_MIN) + DB_MIN
    return spec_db


def pad_spectrogram(spec, target_frames=TARGET_FRAMES, target_freq=TARGET_FREQ):
    """
    Pad spectrogram to target size (no information loss).
    
    Args:
        spec: Spectrogram [freq, time]
        target_frames: Target time dimension
        target_freq: Target frequency dimension
    
    Returns:
        Padded spectrogram, original_shape for later cropping
    """
    original_shape = spec.shape
    freq, time = spec.shape[-2], spec.shape[-1]
    
    # Calculate padding needed
    pad_time = max(0, target_frames - time)
    pad_freq = max(0, target_freq - freq)
    
    # Pad with 0 (which is silence in normalized [0,1] space, i.e. DB_MINdB)
    if spec.dim() == 2:
        spec_padded = F.pad(spec, (0, pad_time, 0, pad_freq), mode='constant', value=0)
    else:
        spec_padded = F.pad(spec, (0, pad_time, 0, pad_freq), mode='constant', value=0)
    
    # Crop if larger than target
    if time > target_frames:
        spec_padded = spec_padded[..., :target_frames]
    if freq > target_freq:
        spec_padded = spec_padded[..., :target_freq, :]
    
    return spec_padded, original_shape


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
        return spec[..., :orig_freq, :orig_time]
    elif len(original_shape) == 3:
        _, orig_freq, orig_time = original_shape
        return spec[..., :orig_freq, :orig_time]
    elif len(original_shape) == 4:
        _, _, orig_freq, orig_time = original_shape
        return spec[..., :orig_freq, :orig_time]
    return spec


def process_audio_for_model(audio, sr=16000):
    """
    Full pipeline: audio -> model-ready spectrogram.
    
    Args:
        audio: Audio tensor [samples] or [channels, samples]
        sr: Sample rate (for reference)
    
    Returns:
        spec_norm: Normalized spectrogram ready for model [1, F, T]
        phase: Phase for reconstruction
        ref: Reference magnitude for reconstruction
        original_shape: For unpadding later
    """
    # Get dB spectrogram
    spec_db, phase, ref = audio_to_spectrogram(audio)
    
    # Normalize to [0, 1]
    spec_norm = normalize_db_spectrogram(spec_db)
    
    # Pad to fixed size
    spec_padded, original_shape = pad_spectrogram(spec_norm)
    
    # Add channel dimension [F, T] -> [1, F, T]
    if spec_padded.dim() == 2:
        spec_padded = spec_padded.unsqueeze(0)
    
    return spec_padded, phase, ref, original_shape


def model_output_to_audio(spec_norm, phase, ref, original_shape):
    """
    Full pipeline: model output -> audio.
    
    Args:
        spec_norm: Model output spectrogram [1, F, T] or [F, T], normalized [0, 1]
        phase: Original phase spectrogram
        ref: Reference magnitude from original audio
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
    
    # Convert to audio using reference
    audio = spectrogram_to_audio(spec_db, phase, ref)
    
    return audio
