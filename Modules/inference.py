from unet import UNet
from audio_utils import (
    audio_to_spectrogram,
    spectrogram_to_audio,
    normalize_db_spectrogram,
    denormalize_db_spectrogram,
    pad_spectrogram,
    unpad_spectrogram,
    N_FFT, HOP_LENGTH, TARGET_FRAMES, TARGET_FREQ, DB_MIN, DB_MAX
)
from pathlib import Path
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm


def dereverb_audio(
    input_audio_path,
    output_audio_path,
    model_path,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Remove reverb from audio file (loads model each time)
    """
    print(f"Using device: {device}")
    print(f"Input: {input_audio_path}")
    print(f"Output: {output_audio_path}")
    
    print("\n[1/5] Loading model...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if 'model_config' in checkpoint:
        model = UNet(**checkpoint['model_config']).to(device)
    else:
        model = UNet(in_channels=1, out_channels=1).to(device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✓ Loaded from epoch {checkpoint.get('epoch', '?')}")
        if 'val_loss' in checkpoint:
            print(f"  ✓ Validation loss: {checkpoint['val_loss']:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("  ✓ Loaded model weights")
    
    model.eval()
    
    clean_audio, sr = _process_audio(
        input_audio_path, output_audio_path, model, device, verbose=True
    )
    
    print("\n" + "="*60)
    print("✓ DEREVERBERATION COMPLETE!")
    print("="*60)
    
    return clean_audio, sr


def dereverb_audio_with_model(
    input_audio_path,
    output_audio_path,
    model,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    verbose=True
):
    """
    Remove reverb from audio file (model already loaded)
    """
    if verbose:
        print(f"Processing: {Path(input_audio_path).name}")
    
    clean_audio, sr = _process_audio(
        input_audio_path, 
        output_audio_path, 
        model, 
        device,
        verbose=verbose
    )
    
    return clean_audio, sr


def dereverb_batch(
    input_dir,
    output_dir,
    model_path,
    file_extension=None,
    output_format='match',
    device='cuda' if torch.cuda.is_available() else 'cpu',
    verbose=True
):
    """
    Process multiple audio files (loads model once)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print("Scanning for audio files...")
    if file_extension is None:
        audio_files = []
        for ext in ['*.wav', '*.flac', '*.mp3', '*.ogg']:
            audio_files.extend(list(input_path.glob(ext)))
        
        if len(audio_files) == 0:
            print(f"No audio files found in {input_dir}")
            return 0, 0
    else:
        audio_files = list(input_path.glob(file_extension))
        
        if len(audio_files) == 0:
            print(f"No files matching {file_extension} found in {input_dir}")
            return 0, 0
    
    print(f"Found {len(audio_files)} audio files in {input_dir}")
    print(f"Output format: {output_format}")
    print(f"Pipeline: dB with ref, n_fft={N_FFT}, hop={HOP_LENGTH}")
    print("="*60)
    
    print(f"\nLoading model on {device}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if 'model_config' in checkpoint:
        model = UNet(**checkpoint['model_config']).to(device)
    else:
        model = UNet(in_channels=1, out_channels=1).to(device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch')
        print(f"✓ Loaded from epoch {epoch + 1 if epoch is not None else '?'}")
        if 'val_loss' in checkpoint:
            print(f"✓ Validation loss: {checkpoint['val_loss']:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("✓ Loaded model weights")
    
    model.eval()
    print("✓ Model ready for batch processing!\n")
    print("="*60)
    
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    pbar = tqdm(audio_files, desc="Processing audio", unit="files", ncols=100)
    
    for audio_file in pbar:
        pbar.set_description(f"Processing: {audio_file.name[:30]}")
        
        if output_format == 'match':
            output_ext = audio_file.suffix
        elif output_format == 'flac':
            output_ext = '.flac'
        elif output_format == 'wav':
            output_ext = '.wav'
        else:
            raise ValueError(f"Invalid output_format: {output_format}")
        
        output_file = output_path / f"clean_{audio_file.stem}{output_ext}"
        
        if output_file.exists():
            skipped_count += 1
            success_count += 1
            pbar.set_postfix({'success': success_count, 'errors': error_count, 'skipped': skipped_count})
            continue
        
        try:
            dereverb_audio_with_model(
                audio_file,
                output_file,
                model,
                device=device,
                verbose=False
            )
            success_count += 1
            pbar.set_postfix({'success': success_count, 'errors': error_count, 'skipped': skipped_count})
        
        except Exception as e:
            error_count += 1
            pbar.set_postfix({'success': success_count, 'errors': error_count, 'skipped': skipped_count})
            tqdm.write(f"  ✗ ERROR in {audio_file.name}: {str(e)[:50]}")
            continue
    
    pbar.close()
    
    print("\n" + "="*60)
    print("BATCH PROCESSING COMPLETE")
    print(f"  Successful: {success_count}/{len(audio_files)}")
    print(f"  Skipped (already exists): {skipped_count}")
    print(f"  Errors: {error_count}/{len(audio_files)}")
    print(f"  Output directory: {output_dir}")
    print("="*60)
    
    return success_count, error_count


def _process_audio(input_path, output_path, model, device, verbose=True):
    """Process audio using dB spectrogram pipeline with reference scaling"""
    
    if verbose:
        print("[2/5] Loading audio...")
    audio, sr = torchaudio.load(str(input_path))
    
    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    
    if verbose:
        print(f"[3/5] Converting to dB spectrogram (n_fft={N_FFT}, hop={HOP_LENGTH})...")
    
    # Get dB spectrogram, phase, and reference
    magnitude_db, phase, ref = audio_to_spectrogram(audio)
    
    if verbose:
        print(f"  Spectrogram shape: {magnitude_db.shape}")
        print(f"  Phase shape: {phase.shape}")
        print(f"  dB range: [{magnitude_db.min():.1f}, {magnitude_db.max():.1f}]")
        print(f"  Reference: {ref:.6f}")
    
    # Normalize to [0, 1]
    magnitude_norm = normalize_db_spectrogram(magnitude_db)
    
    # Pad to model input size
    magnitude_padded, pad_info = pad_spectrogram(
        magnitude_norm, 
        target_frames=TARGET_FRAMES, 
        target_freq=TARGET_FREQ
    )
    
    # Add batch and channel dims [F, T] -> [1, 1, F, T]
    model_input = magnitude_padded.unsqueeze(0).unsqueeze(0).to(device)
    
    if verbose:
        print(f"[4/5] Processing through U-Net...")
        print(f"  Model input: {model_input.shape}")
    
    with torch.no_grad():
        clean_padded = model(model_input).cpu().squeeze(0).squeeze(0)
    
    if verbose:
        print(f"  Model output: {clean_padded.shape}")
        print(f"  Output range (raw): [{clean_padded.min():.3f}, {clean_padded.max():.3f}]")
    
    # Handle NaN/Inf
    if torch.isnan(clean_padded).any() or torch.isinf(clean_padded).any():
        if verbose:
            print("  ⚠️ Replacing NaN/Inf values")
        clean_padded = torch.nan_to_num(clean_padded, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Clamp to valid range
    clean_padded = torch.clamp(clean_padded, 0, 1)
    
    # Unpad to original size
    clean_norm = unpad_spectrogram(clean_padded, pad_info)
    
    if verbose:
        print(f"  After unpad: {clean_norm.shape}")
        print(f"  Phase shape: {phase.shape}")
    
    # Ensure shapes match exactly
    if clean_norm.shape != phase.shape:
        if verbose:
            print(f"  ⚠️ Shape mismatch! Cropping...")
        min_freq = min(clean_norm.shape[-2], phase.shape[-2])
        min_time = min(clean_norm.shape[-1], phase.shape[-1])
        clean_norm = clean_norm[..., :min_freq, :min_time]
        phase = phase[..., :min_freq, :min_time]
    
    if verbose:
        print(f"  Final norm range: [{clean_norm.min():.3f}, {clean_norm.max():.3f}]")
    
    # Denormalize back to dB
    clean_db = denormalize_db_spectrogram(clean_norm)
    
    if verbose:
        print(f"[5/5] Converting back to audio...")
        print(f"  dB range: [{clean_db.min():.1f}, {clean_db.max():.1f}]")
    
    # Convert to audio using the reference from input
    clean_audio = spectrogram_to_audio(clean_db, phase, ref)
    
    # Length matching
    original_length = audio.shape[-1]
    if clean_audio.shape[-1] > original_length:
        clean_audio = clean_audio[..., :original_length]
    elif clean_audio.shape[-1] < original_length:
        padding = original_length - clean_audio.shape[-1]
        clean_audio = F.pad(clean_audio, (0, padding))
    
    # Gentle normalization - only if clipping
    max_val = torch.max(torch.abs(clean_audio))
    if max_val > 1.0:
        clean_audio = clean_audio / max_val * 0.99
        if verbose:
            print(f"  Audio normalized: peak was {max_val:.2f}")
    
    clean_audio = torch.clamp(clean_audio, -1.0, 1.0)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if clean_audio.dim() == 1:
        clean_audio = clean_audio.unsqueeze(0)
    
    torchaudio.save(str(output_path), clean_audio, sr)
    
    if verbose:
        print(f"  ✓ Saved to: {output_path.name}")
    
    return clean_audio, sr


if __name__ == '__main__':
    dereverb_batch(
        input_dir='/scratch/egbueze.m/reverb_dataset/reverb',
        output_dir='/scratch/egbueze.m/reverb_dataset/clean_output',
        model_path='/scratch/egbueze.m/checkpoints_db2/best_model.pth',
        file_extension=None,
        output_format='wav',
        device='cuda'
    )
