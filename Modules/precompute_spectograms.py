import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
from audio_utils import audio_to_spectrogram, resize_spectrogram
import numpy as np
import time


def precompute_all_spectrograms(data_dir, output_dir, normalize=True, checkpoint_interval=100):
    """
    Precompute spectrograms with normalization and progress tracking
    
    Args:
        data_dir: Path to dataset with reverb/ and clean/ subdirectories
        output_dir: Where to save precomputed spectrograms
        normalize: Whether to apply global normalization
        checkpoint_interval: Save checkpoint every N files
    """
    reverb_dir = Path(data_dir) / 'reverb'
    clean_dir = Path(data_dir) / 'clean'
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("Scanning directories...")
    reverb_files = sorted(list(reverb_dir.glob('*.flac')))
    clean_files = sorted(list(clean_dir.glob('*.flac')))
    
    assert len(reverb_files) == len(clean_files), f"Mismatch: {len(reverb_files)} reverb, {len(clean_files)} clean"
    
    total_files = len(reverb_files)
    print(f"Found {total_files} audio file pairs")
    print("="*70)
    
    if normalize:
        print("\n📊 PASS 1/2: Collecting statistics for normalization")
        print("-"*70)
        
        all_max_values = []
        
        # First pass - collect all max values with progress bar
        pbar = tqdm(
            zip(reverb_files, clean_files),
            total=total_files,
            desc="Analyzing",
            unit="pairs",
            ncols=100
        )
        
        for reverb_file, clean_file in pbar:
            # Update with current file
            pbar.set_description(f"Analyzing: {reverb_file.stem[:25]}")
            
            try:
                reverb_audio, _ = torchaudio.load(str(reverb_file))
                clean_audio, _ = torchaudio.load(str(clean_file))
                
                reverb_mag, _ = audio_to_spectrogram(reverb_audio, n_fft=512, hop_length=256)
                clean_mag, _ = audio_to_spectrogram(clean_audio, n_fft=512, hop_length=256)
                
                reverb_spec = resize_spectrogram(reverb_mag, target_size=(256, 256)).squeeze(0)
                clean_spec = resize_spectrogram(clean_mag, target_size=(256, 256)).squeeze(0)
                
                max_val = max(reverb_spec.max().item(), clean_spec.max().item())
                all_max_values.append(max_val)
                
                # Update stats in postfix
                if len(all_max_values) % 50 == 0:
                    current_99 = np.percentile(all_max_values, 99)
                    pbar.set_postfix({'99%ile': f'{current_99:.2f}'})
                    
            except Exception as e:
                tqdm.write(f"⚠️ Error in {reverb_file.name}: {str(e)[:50]}")
                continue
        
        pbar.close()
        
        # Calculate normalization statistics
        global_norm = np.percentile(all_max_values, 99)
        
        print(f"\n✅ Statistics collected from {len(all_max_values)} files:")
        print(f"   Max value: {np.max(all_max_values):.2f}")
        print(f"   Mean max: {np.mean(all_max_values):.2f} ± {np.std(all_max_values):.2f}")
        print(f"   99th percentile (used for norm): {global_norm:.2f}")
        
        print("\n🔄 PASS 2/2: Processing and saving spectrograms")
        print("-"*70)
    else:
        all_max_values = None
        global_norm = None
        print("\n🔄 Processing spectrograms (no normalization)")
        print("-"*70)
    
    # Check for existing files to skip
    existing_files = list(output_dir.glob('spec_*.pt'))
    start_idx = len(existing_files)
    
    if start_idx > 0:
        print(f"📂 Found {start_idx} existing spectrograms, resuming from index {start_idx}")
    
    # Main processing pass with detailed progress
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
    
    for idx, (reverb_file, clean_file in pbar:
        # Skip if already processed
        output_file = output_dir / f'spec_{idx:06d}.pt'
        if output_file.exists():
            processed_count += 1
            continue
        
        # Update description
        pbar.set_description(f"Processing: {reverb_file.stem[:25]}")
        
        try:
            reverb_audio, _ = torchaudio.load(str(reverb_file))
            clean_audio, _ = torchaudio.load(str(clean_file))
            
            reverb_mag, _ = audio_to_spectrogram(reverb_audio, n_fft=512, hop_length=256)
            clean_mag, _ = audio_to_spectrogram(clean_audio, n_fft=512, hop_length=256)
            
            reverb_spec = resize_spectrogram(reverb_mag, target_size=(256, 256)).squeeze(0)
            clean_spec = resize_spectrogram(clean_mag, target_size=(256, 256)).squeeze(0)
            
            # Store original max for reference
            original_max = max(reverb_spec.max().item(), clean_spec.max().item())
            
            # ============ NORMALIZATION ============
            if normalize:
                # USE GLOBAL NORM FOR ALL FILES
                reverb_spec = reverb_spec / (global_norm + 1e-7)
                clean_spec = clean_spec / (global_norm + 1e-7)
                
                # Ensure [0, 1] range
                reverb_spec = torch.clamp(reverb_spec, 0, 1)
                clean_spec = torch.clamp(clean_spec, 0, 1)
                
                max_val = global_norm  # Store the global norm used
            else:
                max_val = None
            # ========================================
            
            torch.save({
                'reverb': reverb_spec,
                'clean': clean_spec,
                'reverb_file': reverb_file.name,
                'clean_file': clean_file.name, 
                'normalized': normalize,
                'max_val': max_val if normalize else None,
                'original_max': original_max  # Keep track of original for debugging
            }, output_file)
            
            processed_count += 1
            
            # Update stats every 100 files
            if processed_count % 100 == 0:
                elapsed = time.time() - start_time
                rate = processed_count / elapsed
                eta = (total_files - idx) / rate if rate > 0 else 0
                
                pbar.set_postfix({
                    'saved': processed_count,
                    'errors': error_count,
                    'rate': f'{rate:.1f}/s',
                    'eta': f'{eta/60:.1f}m'
                })
                
        except Exception as e:
            error_count += 1
            tqdm.write(f"⚠️ Error processing {reverb_file.name}: {str(e)[:50]}")
            pbar.set_postfix({'saved': processed_count, 'errors': error_count})
            continue
    
    pbar.close()
    
    # Save normalization statistics
    if normalize:
        stats = {
            'normalized': True,
            'method': 'global_percentile',
            'global_max': np.max(all_max_values),
            'global_mean_max': np.mean(all_max_values),
            'global_std_max': np.std(all_max_values),
            'percentile_99': np.percentile(all_max_values, 99),
            'percentile_95': np.percentile(all_max_values, 95),
            'percentile_90': np.percentile(all_max_values, 90),
            'global_norm_used': global_norm,
            'total_files': len(all_max_values)
        }
        torch.save(stats, output_dir / 'normalization_stats.pt')
        
        print("\n📈 Final normalization statistics saved:")
        print(f"   Files analyzed: {stats['total_files']}")
        print(f"   Global max: {stats['global_max']:.2f}")
        print(f"   Mean max: {stats['global_mean_max']:.2f} ± {stats['global_std_max']:.2f}")
        print(f"   Percentiles: 90th={stats['percentile_90']:.2f}, "
              f"95th={stats['percentile_95']:.2f}, 99th={stats['percentile_99']:.2f}")
        print(f"   Normalization value: {global_norm:.2f}")
    
    # Final summary
    elapsed_total = time.time() - start_time
    print("\n" + "="*70)
    print("✅ PRECOMPUTATION COMPLETE")
    print("="*70)
    print(f"   Total processed: {processed_count}/{total_files}")
    print(f"   Errors: {error_count}")
    print(f"   Time elapsed: {elapsed_total/60:.1f} minutes")
    print(f"   Average rate: {processed_count/elapsed_total:.1f} files/second")
    print(f"   Output directory: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    precompute_all_spectrograms(
        data_dir='/scratch/egbueze.m/reverb_dataset',
        output_dir='/scratch/egbueze.m/precomputed_specs_normalized',
        normalize=True
    )
