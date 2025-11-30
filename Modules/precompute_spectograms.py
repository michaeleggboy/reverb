from audio_utils import (
    audio_to_spectrogram,
    normalize_db_spectrogram,
    pad_spectrogram,
    N_FFT, HOP_LENGTH, TARGET_FRAMES, TARGET_FREQ, DB_MIN, DB_MAX
)
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
import numpy as np
import time


def precompute_spectrograms(data_dir, output_dir):
    """
    Precompute dB-normalized spectrograms.
    
    Args:
        data_dir: Path to dataset with reverb/ and clean/ subdirectories
        output_dir: Where to save precomputed spectrograms
    """
    reverb_dir = Path(data_dir) / 'reverb'
    clean_dir = Path(data_dir) / 'clean'
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*70)
    print("SPECTROGRAM PRECOMPUTATION V2 (dB-scale)")
    print("="*70)
    print(f"  n_fft:         {N_FFT}")
    print(f"  hop_length:    {HOP_LENGTH}")
    print(f"  target_frames: {TARGET_FRAMES}")
    print(f"  target_freq:   {TARGET_FREQ}")
    print(f"  dB range:      [{DB_MIN}, {DB_MAX}]")
    print("="*70)
    
    # Scan files
    print("\nScanning directories...")
    reverb_files = sorted(list(reverb_dir.glob('*.flac')))
    clean_files = sorted(list(clean_dir.glob('*.flac')))
    
    assert len(reverb_files) == len(clean_files), \
        f"Mismatch: {len(reverb_files)} reverb, {len(clean_files)} clean"
    
    total_files = len(reverb_files)
    print(f"Found {total_files} audio file pairs")
    
    # Check for existing files to resume
    existing_files = list(output_dir.glob('spec_*.pt'))
    start_idx = len(existing_files)
    
    if start_idx > 0:
        print(f"📂 Found {start_idx} existing spectrograms, resuming...")
    
    # Stats collection
    all_differences = []
    freq_dims = []
    time_dims = []
    
    # Processing
    pbar = tqdm(
        enumerate(zip(reverb_files, clean_files)),
        total=total_files,
        initial=start_idx,
        desc="Processing",
        unit="pairs",
        ncols=100
    )
    
    processed_count = 0
    error_count = 0
    start_time = time.time()
    
    for idx, (reverb_file, clean_file) in pbar:
        output_file = output_dir / f'spec_{idx:06d}.pt'
        
        # Skip if already processed
        if output_file.exists():
            continue
        
        try:
            # Load audio
            reverb_audio, sr = torchaudio.load(str(reverb_file))
            clean_audio, _ = torchaudio.load(str(clean_file))
            
            # Convert to dB spectrograms
            reverb_db, reverb_phase = audio_to_spectrogram(reverb_audio, return_db=True)
            clean_db, clean_phase = audio_to_spectrogram(clean_audio, return_db=True)
            
            # Track original dimensions (first few files)
            if len(freq_dims) < 10:
                freq_dims.append(reverb_db.shape[0])
                time_dims.append(reverb_db.shape[1])
            
            # Normalize to [0, 1]
            reverb_norm = normalize_db_spectrogram(reverb_db)
            clean_norm = normalize_db_spectrogram(clean_db)
            
            # Track differences (for stats)
            diff = torch.abs(reverb_norm - clean_norm).mean().item()
            all_differences.append(diff)
            
            # Pad to fixed size
            reverb_padded, orig_shape = pad_spectrogram(reverb_norm, target_frames=TARGET_FRAMES, target_freq=TARGET_FREQ)
            clean_padded, _ = pad_spectrogram(clean_norm, target_frames=TARGET_FRAMES, target_freq=TARGET_FREQ)
            
            # Save
            torch.save({
                'reverb': reverb_padded,
                'clean': clean_padded,
                'reverb_phase': reverb_phase,  # Keep phase for reconstruction
                'clean_phase': clean_phase,
                'original_shape': orig_shape,
                'reverb_file': reverb_file.name,
                'clean_file': clean_file.name,
                'sample_rate': sr,
                # Metadata for inference
                'n_fft': N_FFT,
                'hop_length': HOP_LENGTH,
                'db_min': DB_MIN,
                'db_max': DB_MAX,
            }, output_file)
            
            processed_count += 1
            
            # Update progress
            if processed_count % 100 == 0:
                avg_diff = np.mean(all_differences[-100:]) if all_differences else 0
                pbar.set_postfix({
                    'saved': processed_count,
                    'avg_diff': f'{avg_diff:.4f}',
                    'errors': error_count
                })
                
        except Exception as e:
            error_count += 1
            tqdm.write(f"⚠️ Error: {reverb_file.name}: {str(e)[:50]}")
            continue
    
    pbar.close()
    
    # Save metadata
    if all_differences:
        stats = {
            'version': 2,
            'n_fft': N_FFT,
            'hop_length': HOP_LENGTH,
            'target_frames': TARGET_FRAMES,
            'db_min': DB_MIN,
            'db_max': DB_MAX,
            'total_files': processed_count,
            'mean_difference': float(np.mean(all_differences)),
            'std_difference': float(np.std(all_differences)),
            'max_difference': float(np.max(all_differences)),
            'freq_dim': int(np.mean(freq_dims)) if freq_dims else None,
            'time_dim_mean': float(np.mean(time_dims)) if time_dims else None,
        }
        torch.save(stats, output_dir / 'preprocessing_stats.pt')
        
        print("\n" + "="*70)
        print("PRECOMPUTATION COMPLETE")
        print("="*70)
        print(f"  Total processed: {processed_count}/{total_files}")
        print(f"  Errors: {error_count}")
        print(f"  Time: {(time.time() - start_time)/60:.1f} minutes")
        print(f"\n📊 SPECTROGRAM STATISTICS:")
        print(f"  Mean L1 difference: {stats['mean_difference']:.4f}")
        print(f"  Std L1 difference:  {stats['std_difference']:.4f}")
        print(f"  Max L1 difference:  {stats['max_difference']:.4f}")
        print(f"  Frequency bins:     {stats['freq_dim']}")
        print(f"  Avg time frames:    {stats['time_dim_mean']:.0f}")
        
        if stats['mean_difference'] > 0.02:
            print("\n  ✅ Good separation between reverb and clean!")
        elif stats['mean_difference'] > 0.01:
            print("\n  🟡 Moderate separation - should be learnable")
        else:
            print("\n  🔴 Low separation - may need stronger reverb")
        
        print("="*70)


def verify_preprocessing(output_dir, num_samples=10):
    """
    Verify the preprocessed spectrograms look correct.
    """
    output_dir = Path(output_dir)
    spec_files = sorted(list(output_dir.glob('spec_*.pt')))[:num_samples]
    
    print(f"\nVerifying {len(spec_files)} samples...")
    
    for spec_file in spec_files:
        data = torch.load(spec_file)
        reverb = data['reverb']
        clean = data['clean']
        
        diff = torch.abs(reverb - clean).mean().item()
        
        print(f"  {spec_file.name}: shape={reverb.shape}, "
              f"range=[{reverb.min():.3f}, {reverb.max():.3f}], "
              f"diff={diff:.4f}")
    
    # Load stats
    stats_file = output_dir / 'preprocessing_stats.pt'
    if stats_file.exists():
        stats = torch.load(stats_file)
        print(f"\n  Overall mean diff: {stats['mean_difference']:.4f}")


if __name__ == '__main__':
    precompute_spectrograms(
        data_dir='/scratch/egbueze.m/reverb_dataset',
        output_dir='/scratch/egbueze.m/precomputed_specs_normalized'    
    )
    
    # Verify
    verify_preprocessing('/scratch/egbueze.m/precomputed_specs_normalized')
