import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
from audio_utils import audio_to_spectrogram, resize_spectrogram
import numpy as np


def precompute_all_spectrograms(data_dir, output_dir, normalize=True):
    reverb_dir = Path(data_dir) / 'reverb'
    clean_dir = Path(data_dir) / 'clean'
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    reverb_files = sorted(list(reverb_dir.glob('*.flac')))
    clean_files = sorted(list(clean_dir.glob('*.flac')))
    
    assert len(reverb_files) == len(clean_files), f"Mismatch: {len(reverb_files)} reverb, {len(clean_files)} clean"
    print(f"Pre-computing {len(reverb_files)} spectrograms...")
    
    if normalize:
        print("Pass 1: Finding global normalization value...")
        all_max_values = []
        
        # First pass - collect all max values
        for reverb_file, clean_file in tqdm(zip(reverb_files, clean_files)):
            reverb_audio, _ = torchaudio.load(str(reverb_file))
            clean_audio, _ = torchaudio.load(str(clean_file))
            
            reverb_mag, _ = audio_to_spectrogram(reverb_audio, n_fft=512, hop_length=256)
            clean_mag, _ = audio_to_spectrogram(clean_audio, n_fft=512, hop_length=256)
            
            reverb_spec = resize_spectrogram(reverb_mag, target_size=(256, 256)).squeeze(0)
            clean_spec = resize_spectrogram(clean_mag, target_size=(256, 256)).squeeze(0)
            
            max_val = max(reverb_spec.max().item(), clean_spec.max().item())
            all_max_values.append(max_val)
        
        # Use 99th percentile as global norm
        global_norm = np.percentile(all_max_values, 99)
        print(f"Global normalization value: {global_norm:.2f}")
        
        print("Pass 2: Processing and normalizing...")
    else:
        all_max_values = None
        global_norm = None
        print("Processing without normalization...")
    
    # Main processing pass
    for idx, (reverb_file, clean_file) in enumerate(tqdm(zip(reverb_files, clean_files))):
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
        }, output_dir / f'spec_{idx:06d}.pt')
    
    # Save normalization statistics
    if normalize:
        stats = {
            'normalized': True,
            'method': 'global_percentile',
            'global_max': np.max(all_max_values),
            'global_mean_max': np.mean(all_max_values),
            'global_std_max': np.std(all_max_values),
            'percentile_99': np.percentile(all_max_values, 99),
            'global_norm_used': global_norm
        }
        torch.save(stats, output_dir / 'normalization_stats.pt')
        
        print(f"✅ Normalization stats:")
        print(f"   Global max: {stats['global_max']:.2f}")
        print(f"   Mean max: {stats['global_mean_max']:.2f}")
        print(f"   99th percentile: {stats['percentile_99']:.2f}")
        print(f"   Norm value used: {global_norm:.2f}")
    
    print(f"✅ Saved {len(reverb_files)} spectrograms to {output_dir}")


if __name__ == '__main__':
    precompute_all_spectrograms(
        data_dir='/scratch/egbueze.m/reverb_dataset',
        output_dir='/scratch/egbueze.m/precomputed_specs_normalized',
        normalize=True
    )
