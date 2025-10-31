import torch
import torch.nn.functional as F
import torchaudio
from pathlib import Path
from model import UNet
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
    Remove reverb from audio file
    
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
    
    # ===== Load Model =====
    print("\n[1/6] Loading model...")
    model = UNet(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✓ Loaded from epoch {checkpoint.get('epoch', '?')}")
        if 'val_loss' in checkpoint:
            print(f"  ✓ Validation loss: {checkpoint['val_loss']:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print(f"  ✓ Loaded model weights")
    
    model.eval()
    
    # ===== Load Audio =====
    print("\n[2/6] Loading audio...")
    audio, sr = torchaudio.load(str(input_audio_path))
    print(f"  ✓ Shape: {audio.shape}, Sample rate: {sr} Hz")
    
    # Convert stereo to mono
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
        print(f"  ✓ Converted to mono: {audio.shape}")
    
    # ===== Convert to Spectrogram =====
    print("\n[3/6] Converting to spectrogram...")
    magnitude, phase = audio_to_spectrogram(audio)
    original_size = magnitude.shape[-2:]
    print(f"  ✓ Original spectrogram: {magnitude.shape}")
    print(f"  ✓ Saved phase for reconstruction")
    
    # ===== Resize for Model =====
    print("\n[4/6] Preparing for model...")
    magnitude_resized = resize_spectrogram(magnitude, target_size=(256, 256))
    print(f"  ✓ Resized to: {magnitude_resized.shape}")
    
    # ===== Run Through Model =====
    print("\n[5/6] Processing through U-Net...")
    with torch.no_grad():
        magnitude_resized = magnitude_resized.to(device)
        clean_magnitude_resized = model(magnitude_resized)
        clean_magnitude_resized = clean_magnitude_resized.cpu()
    print(f"  ✓ Predicted clean spectrogram")
    
    # Resize back to original dimensions
    clean_magnitude = unresize_spectrogram(clean_magnitude_resized, original_size)
    clean_magnitude = clean_magnitude.squeeze(0)  # Remove batch dimension
    print(f"  ✓ Resized back to: {clean_magnitude.shape}")
    
    # ===== Convert Back to Audio =====
    print("\n[6/6] Converting back to audio...")
    # Use original phase with predicted clean magnitude
    clean_audio = spectrogram_to_audio(clean_magnitude, phase)
    print(f"  ✓ Audio shape: {clean_audio.shape}")
    
    # Match length to input
    if clean_audio.shape[-1] > audio.shape[-1]:
        clean_audio = clean_audio[..., :audio.shape[-1]]
    elif clean_audio.shape[-1] < audio.shape[-1]:
        padding = audio.shape[-1] - clean_audio.shape[-1]
        clean_audio = F.pad(clean_audio, (0, padding))
    
    # Normalize to prevent clipping
    max_val = torch.max(torch.abs(clean_audio))
    if max_val > 1.0:
        clean_audio = clean_audio / max_val * 0.99
        print(f"  ✓ Normalized (peak was {max_val:.2f})")
    
    # ===== Save Output =====
    print("\nSaving clean audio...")
    # Ensure output directory exists
    output_path = Path(output_audio_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save (add channel dimension if needed)
    if clean_audio.dim() == 1:
        clean_audio = clean_audio.unsqueeze(0)
    
    torchaudio.save(str(output_audio_path), clean_audio, sr)
    print(f"✓ Saved to: {output_audio_path}")
    
    print("\n" + "="*60)
    print("✓ DEREVERBERATION COMPLETE!")
    print("="*60)
    
    return clean_audio, sr


def dereverb_batch(
    input_dir,
    output_dir,
    model_path,
    file_extension='*.wav',
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Process multiple audio files in a directory
    
    Args:
        input_dir: Directory containing reverberant audio files
        output_dir: Directory to save clean audio files
        model_path: Path to trained model
        file_extension: Pattern to match (e.g., '*.wav', '*.flac')
        device: 'cuda' or 'cpu'
    """
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Get all matching audio files
    audio_files = list(input_path.glob(file_extension))
    
    # Also check for other common formats
    if file_extension == '*.wav':
        audio_files += list(input_path.glob('*.flac'))
        audio_files += list(input_path.glob('*.mp3'))
    
    print(f"Found {len(audio_files)} audio files in {input_dir}")
    print("="*60)
    
    success_count = 0
    error_count = 0
    
    for i, audio_file in enumerate(audio_files):
        print(f"\n[{i+1}/{len(audio_files)}] Processing: {audio_file.name}")
        print("-"*60)
        
        # Output filename: clean_originalname.wav
        output_file = output_path / f"clean_{audio_file.stem}.wav"
        
        try:
            dereverb_audio(audio_file, output_file, model_path, device)
            success_count += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            error_count += 1
            continue
    
    print("\n" + "="*60)
    print("BATCH PROCESSING COMPLETE")
    print(f"  Successful: {success_count}/{len(audio_files)}")
    print(f"  Errors: {error_count}/{len(audio_files)}")
    print("="*60)

if __name__ == '__main__':
    
    dereverb_audio(
        input_audio_path='reverberant_speech.wav',
        output_audio_path='clean_speech.wav',
        model_path='checkpoints/best_model.pth',
        device='cuda'  # or 'cpu'
    )
    
    # Example 2: Process all files in a directory
    # dereverb_batch(
    #     input_dir='./test_audio',
    #     output_dir='./clean_output',
    #     model_path='checkpoints/best_model.pth',
    #     device='cuda'
    # )
    