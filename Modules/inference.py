from pathlib import Path
import torch
import torch.nn.functional as F
import torchaudio
from unet import UNet
from audio_utils import (
    audio_to_spectrogram,
    spectrogram_to_audio,
    resize_spectrogram,
    unresize_spectrogram
)


def dereverb_audio(
    input_audio_path,
    output_audio_path,
    model_path,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Remove reverb from audio file (loads model each time)
    
    Args:
        input_audio_path: Path to reverberant audio (.wav, .flac, .mp3)
        output_audio_path: Path to save clean audio
        model_path: Path to trained model checkpoint (.pth)
        device: 'cuda' or 'cpu'
    
    Returns:
        clean_audio: Processed audio tensor
        sr: Sample rate
    """
    
    print(f"Using device: {device}")
    print(f"Input: {input_audio_path}")
    print(f"Output: {output_audio_path}")
    
    # Load model
    print("\n[1/6] Loading model...")
    model = UNet(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✓ Loaded from epoch {checkpoint.get('epoch', '?')}")
        if 'val_loss' in checkpoint:
            print(f"  ✓ Validation loss: {checkpoint['val_loss']:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("  ✓ Loaded model weights")
    
    model.eval()
    
    # Process audio
    clean_audio, sr = _process_audio(input_audio_path, output_audio_path, model, device, verbose=True)
    
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
    
    Args:
        input_audio_path: Path to reverberant audio
        output_audio_path: Path to save clean audio
        model: Pre-loaded UNet model
        device: 'cuda' or 'cpu'
        verbose: Print detailed progress
    
    Returns:
        clean_audio: Processed audio tensor
        sr: Sample rate
    """
    
    if verbose:
        print(f"Processing: {Path(input_audio_path).name}")
    
    # Process audio
    clean_audio, sr = _process_audio(
        input_audio_path, 
        output_audio_path, 
        model, 
        device, 
        verbose
    )
    
    return clean_audio, sr


def _process_audio(input_path, output_path, model, device, verbose=True):
    """Internal function to process audio with pre-loaded model"""
    
    # Load audio
    if verbose:
        print("  Loading audio...")
    audio, sr = torchaudio.load(str(input_path))
    
    # Convert to mono
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    
    # To spectrogram
    if verbose:
        print("  Converting to spectrogram...")
    magnitude, phase = audio_to_spectrogram(audio)
    original_size = magnitude.shape[-2:]
    
    # Resize
    magnitude_resized = resize_spectrogram(magnitude, (256, 256))
    
    # Predict
    if verbose:
        print("  Processing through U-Net...")
    with torch.no_grad():
        clean_resized = model(magnitude_resized.to(device)).cpu()
    
    # Check for invalid values
    if torch.isnan(clean_resized).any():
        if verbose:
            print("  ⚠️ Replacing NaN values")
        clean_resized = torch.nan_to_num(clean_resized, nan=0.0)
    
    if torch.isinf(clean_resized).any():
        if verbose:
            print("  ⚠️ Replacing Inf values")
        clean_resized = torch.nan_to_num(clean_resized, posinf=5.0, neginf=0.0)
    
    # Clamp extreme values
    max_before = clean_resized.max().item()
    clean_resized = torch.clamp(clean_resized, min=0.0, max=5.0)
    
    if verbose and max_before > 5.0:
        print(f"  ⚠️ Clamped max value from {max_before:.2f} to 5.0")
    
    # Unresize
    clean_magnitude = unresize_spectrogram(clean_resized, original_size)
    clean_magnitude = clean_magnitude.squeeze(0)
    
    # Reconstruct
    if verbose:
        print("  Converting back to audio...")
    clean_audio = spectrogram_to_audio(clean_magnitude, phase)
    
    # Match length
    if clean_audio.shape[-1] > audio.shape[-1]:
        clean_audio = clean_audio[..., :audio.shape[-1]]
    elif clean_audio.shape[-1] < audio.shape[-1]:
        padding = audio.shape[-1] - clean_audio.shape[-1]
        clean_audio = F.pad(clean_audio, (0, padding))
    
    # Normalize
    max_val = torch.max(torch.abs(clean_audio))
    if max_val > 1.0:
        clean_audio = clean_audio / max_val * 0.99
    
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
    Process multiple audio files (OPTIMIZED: loads model once)
    
    Args:
        input_dir: Directory containing reverberant audio files
        output_dir: Directory to save clean audio files
        model_path: Path to trained model checkpoint
        file_extension: File pattern (None=all audio, '*.flac', '*.wav', etc.)
        output_format: Output format - 'match', 'flac', or 'wav'
        device: 'cuda' or 'cpu'
        verbose: Print progress
    """
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Get audio files based on extension
    if file_extension is None:
        # Get all common audio formats
        audio_files = []
        for ext in ['*.wav', '*.flac', '*.mp3', '*.ogg']:
            audio_files.extend(list(input_path.glob(ext)))
        
        if len(audio_files) == 0:
            print(f"No audio files found in {input_dir}")
            return
    else:
        # Get specific extension only
        audio_files = list(input_path.glob(file_extension))
        
        if len(audio_files) == 0:
            print(f"No files matching {file_extension} found in {input_dir}")
            return
    
    print(f"Found {len(audio_files)} audio files in {input_dir}")
    print(f"Output format: {output_format}")
    print("="*60)
    
    # Load model once
    print(f"\nLoading model on {device}...")
    model = UNet(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded from epoch {checkpoint.get('epoch', '?')}")
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
    
    # Process each file with pre-loaded model
    for i, audio_file in enumerate(audio_files):
        if verbose:
            print(f"\n[{i+1}/{len(audio_files)}] {audio_file.name}")
        
        # Determine output extension
        if output_format == 'match':
            output_ext = audio_file.suffix
        elif output_format == 'flac':
            output_ext = '.flac'
        elif output_format == 'wav':
            output_ext = '.wav'
        else:
            raise ValueError(f"Invalid output_format: {output_format}. Use 'match', 'flac', or 'wav'")
        
        output_file = output_path / f"clean_{audio_file.stem}{output_ext}"
        
        # Skip if already processed
        if output_file.exists():
            if verbose:
                print("  ⏭️  Skipping (already exists)")
            success_count += 1
            continue
        
        try:
            # Use pre-loaded model
            dereverb_audio_with_model(
                audio_file,
                output_file,
                model,
                device,
                verbose=False
            )
            success_count += 1
            
            if verbose:
                print(f"  ✓ Saved as {output_ext}")
            
            # Progress update every 50 files
            if verbose and (i + 1) % 50 == 0:
                pct = 100 * (i + 1) / len(audio_files)
                print(f"\n📊 Progress: {i+1}/{len(audio_files)} ({pct:.1f}%)")
                print(f"   Success: {success_count}, Errors: {error_count}")
        
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            error_count += 1
            continue
    
    print("\n" + "="*60)
    print("BATCH PROCESSING COMPLETE")
    print(f"  Successful: {success_count}/{len(audio_files)}")
    print(f"  Errors: {error_count}/{len(audio_files)}")
    print("="*60)
    
    return success_count, error_count


if __name__ == '__main__':
    
    # Example 1: Process single file
    dereverb_audio(
        input_audio_path='reverberant_speech.flac',
        output_audio_path='clean_speech.flac',
        model_path='checkpoints/best_model.pth',
        device='cuda'
    )
    
    # Example 2: Process all audio files, save as FLAC
    # dereverb_batch(
    #     input_dir='./reverb',
    #     output_dir='./clean',
    #     model_path='checkpoints/best_model.pth',
    #     file_extension=None,      # All formats
    #     output_format='flac',     # Save as FLAC
    #     device='cuda'
    # )
    
    # Example 3: Process only FLAC files, match format
    # dereverb_batch(
    #     input_dir='./reverb',
    #     output_dir='./clean',
    #     model_path='checkpoints/best_model.pth',
    #     file_extension='*.flac',  # Only FLAC
    #     output_format='match',    # Keep as FLAC
    #     device='cuda'
    # )
    
    # Example 4: Process FLAC files, output as WAV
    # dereverb_batch(
    #     input_dir='./reverb',
    #     output_dir='./clean',
    #     model_path='checkpoints/best_model.pth',
    #     file_extension='*.flac',  # FLAC input
    #     output_format='wav',      # WAV output
    #     device='cuda'
    # )