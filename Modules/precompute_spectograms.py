from audio_utils import (
    audio_to_spectrogram,
    normalize_db_spectrogram,
    pad_spectrogram,
    N_FFT, HOP_LENGTH, TARGET_FRAMES, TARGET_FREQ
)
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
import random


def precompute_spectrograms(
    data_dir,
    output_dir,
    max_samples=None,
    seed=42
):
    """
    Precompute and save spectrogram pairs for training.
    
    Args:
        data_dir: Directory containing 'clean' and 'reverb' subdirectories
        output_dir: Directory to save precomputed spectrograms
        max_samples: Maximum number of samples to process (None for all)
        seed: Random seed for reproducibility
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    clean_dir = data_dir / 'clean'
    reverb_dir = data_dir / 'reverb'
    
    # Find matching pairs
    clean_files = sorted(list(clean_dir.glob('*.flac')) + list(clean_dir.glob('*.wav')))
    
    print(f"Found {len(clean_files)} clean files")
    print(f"Output directory: {output_dir}")
    print(f"Pipeline: dB with ref + phase, n_fft={N_FFT}, hop={HOP_LENGTH}")
    print(f"Target size: {TARGET_FREQ}x{TARGET_FRAMES}")
    print("="*70)
    
    # Shuffle and limit if requested
    if max_samples is not None:
        random.seed(seed)
        random.shuffle(clean_files)
        clean_files = clean_files[:max_samples]
        print(f"Processing {len(clean_files)} samples (max_samples={max_samples})")
    
    processed = 0
    skipped = 0
    errors = 0
    
    # Track statistics
    all_diffs = []
    
    for idx, clean_path in enumerate(tqdm(clean_files, desc="Precomputing spectrograms")):
        # Find matching reverb file
        reverb_path = reverb_dir / clean_path.name
        if not reverb_path.exists():
            reverb_path = reverb_dir / (clean_path.stem + '.wav')
        if not reverb_path.exists():
            reverb_path = reverb_dir / (clean_path.stem + '.flac')
        
        if not reverb_path.exists():
            skipped += 1
            continue
        
        output_path = output_dir / f'spec_{idx:05d}.pt'
        
        # Skip if already exists
        if output_path.exists():
            processed += 1
            continue
        
        try:
            # Load audio
            clean_audio, sr = torchaudio.load(str(clean_path))
            reverb_audio, _ = torchaudio.load(str(reverb_path))
            
            # Convert to mono
            if clean_audio.shape[0] > 1:
                clean_audio = clean_audio.mean(dim=0, keepdim=True)
            if reverb_audio.shape[0] > 1:
                reverb_audio = reverb_audio.mean(dim=0, keepdim=True)
            
            # Get spectrograms with phase and references
            clean_db, clean_phase, clean_ref = audio_to_spectrogram(clean_audio)
            reverb_db, reverb_phase, reverb_ref = audio_to_spectrogram(reverb_audio)
            
            # Normalize to [0, 1]
            clean_norm = normalize_db_spectrogram(clean_db)
            reverb_norm = normalize_db_spectrogram(reverb_db)
            
            # Pad magnitude to target size
            clean_padded, orig_shape = pad_spectrogram(clean_norm, TARGET_FRAMES, TARGET_FREQ)
            reverb_padded, _ = pad_spectrogram(reverb_norm, TARGET_FRAMES, TARGET_FREQ)
            
            # Pad phase to same size (for optional use in training)
            clean_phase_padded, _ = pad_spectrogram(clean_phase, TARGET_FRAMES, TARGET_FREQ)
            reverb_phase_padded, _ = pad_spectrogram(reverb_phase, TARGET_FRAMES, TARGET_FREQ)
            
            # Track difference for statistics
            diff = torch.abs(reverb_padded - clean_padded).mean().item()
            all_diffs.append(diff)
            
            # Save everything
            torch.save({
                'reverb': reverb_padded,
                'clean': clean_padded,
                'reverb_phase': reverb_phase_padded,
                'clean_phase': clean_phase_padded,
                'reverb_ref': reverb_ref,
                'clean_ref': clean_ref,
                'original_shape': orig_shape,
                'sample_rate': sr,
                'clean_file': clean_path.name,
                'reverb_file': reverb_path.name
            }, output_path)
            
            processed += 1
            
        except Exception as e:
            errors += 1
            tqdm.write(f"Error processing {clean_path.name}: {e}")
            continue
    
    # Save statistics
    if all_diffs:
        stats = {
            'version': 2,
            'num_samples': processed,
            'mean_difference': sum(all_diffs) / len(all_diffs),
            'min_difference': min(all_diffs),
            'max_difference': max(all_diffs),
            'n_fft': N_FFT,
            'hop_length': HOP_LENGTH,
            'target_frames': TARGET_FRAMES,
            'target_freq': TARGET_FREQ,
            'includes_phase': True
        }
        torch.save(stats, output_dir / 'preprocessing_stats.pt')
        
        print("\n" + "="*70)
        print("PRECOMPUTATION COMPLETE")
        print(f"  Processed: {processed}")
        print(f"  Skipped (no match): {skipped}")
        print(f"  Errors: {errors}")
        print(f"  Mean reverb-clean diff: {stats['mean_difference']:.4f}")
        print(f"  Includes phase: True")
        print(f"  Output: {output_dir}")
        print("="*70)
    
    return processed


def verify_preprocessing(output_dir, num_samples=5):
    """
    Verify the preprocessed spectrograms and test reconstruction.
    """
    from audio_utils import denormalize_db_spectrogram, spectrogram_to_audio, unpad_spectrogram
    
    output_dir = Path(output_dir)
    spec_files = sorted(list(output_dir.glob('spec_*.pt')))[:num_samples]
    
    print(f"\nVerifying {len(spec_files)} samples...")
    
    for spec_file in spec_files:
        data = torch.load(spec_file, weights_only=False)
        reverb = data['reverb']
        clean = data['clean']
        
        diff = torch.abs(reverb - clean).mean().item()
        
        has_phase = 'reverb_phase' in data and data['reverb_phase'] is not None
        has_refs = 'reverb_ref' in data and 'clean_ref' in data
        
        print(f"  {spec_file.name}:")
        print(f"    Shape: {reverb.shape}")
        print(f"    Reverb range: [{reverb.min():.3f}, {reverb.max():.3f}]")
        print(f"    Clean range: [{clean.min():.3f}, {clean.max():.3f}]")
        print(f"    Difference: {diff:.4f}")
        print(f"    Has phase: {has_phase}")
        print(f"    Has refs: {has_refs}")
    
    # Load stats
    stats_file = output_dir / 'preprocessing_stats.pt'
    if stats_file.exists():
        stats = torch.load(stats_file, weights_only=False)
        print(f"\nDataset statistics:")
        print(f"  Version: {stats.get('version', 1)}")
        print(f"  Samples: {stats['num_samples']}")
        print(f"  Mean diff: {stats['mean_difference']:.4f}")
        print(f"  Includes phase: {stats.get('includes_phase', False)}")
    
    # Test reconstruction on first sample
    print("\nTesting reconstruction...")
    data = torch.load(spec_files[0], weights_only=False)
    
    # Reconstruct clean audio
    clean_norm = data['clean'].squeeze(0)
    clean_phase = data['clean_phase'].squeeze(0) if data.get('clean_phase') is not None else None
    clean_ref = data.get('clean_ref', torch.tensor(1.0))
    orig_shape = data['original_shape']
    
    if clean_phase is not None:
        # Unpad
        clean_norm = unpad_spectrogram(clean_norm, orig_shape)
        clean_phase = unpad_spectrogram(clean_phase, orig_shape)
        
        # Denormalize and reconstruct
        clean_db = denormalize_db_spectrogram(clean_norm)
        clean_audio = spectrogram_to_audio(clean_db, clean_phase, clean_ref)
        
        print(f"  Reconstructed audio shape: {clean_audio.shape}")
        print(f"  Audio range: [{clean_audio.min():.3f}, {clean_audio.max():.3f}]")
        print("  ✓ Reconstruction successful!")
    else:
        print("  ⚠️ No phase stored, skipping reconstruction test")


if __name__ == '__main__':
    # Precompute spectrograms
    precompute_spectrograms(
        data_dir='/scratch/egbueze.m/reverb_dataset',
        output_dir='/scratch/egbueze.m/precomputed_specs_db',
        max_samples=None
    )
    
    # Verify
    verify_preprocessing('/scratch/egbueze.m/precomputed_specs_db')
