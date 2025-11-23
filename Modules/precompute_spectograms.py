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
    
    print(f"Pre-computing {len(reverb_files)} spectrograms...")
    
    # Track global statistics for normalization
    if normalize:
        all_max_values = []
    
    for idx, (reverb_file, clean_file) in enumerate(tqdm(zip(reverb_files, clean_files))):
        reverb_audio, _ = torchaudio.load(str(reverb_file))
        clean_audio, _ = torchaudio.load(str(clean_file))
        
        reverb_mag, _ = audio_to_spectrogram(reverb_audio, n_fft=512, hop_length=256)
        clean_mag, _ = audio_to_spectrogram(clean_audio, n_fft=512, hop_length=256)
        
        reverb_spec = resize_spectrogram(reverb_mag, target_size=(256, 256)).squeeze(0)
        clean_spec = resize_spectrogram(clean_mag, target_size=(256, 256)).squeeze(0)
        
        # ============ NORMALIZATION ============
        if normalize:
            max_val = max(reverb_spec.max().item(), clean_spec.max().item())
            reverb_spec = reverb_spec / (max_val + 1e-7)
            clean_spec = clean_spec / (max_val + 1e-7)
            
            all_max_values.append(max_val)
            
            # Ensure [0, 1] range
            reverb_spec = torch.clamp(reverb_spec, 0, 1)
            clean_spec = torch.clamp(clean_spec, 0, 1)
        # ========================================
        
        torch.save({
            'reverb': reverb_spec,
            'clean': clean_spec,
            'normalized': normalize,
            'max_val': max_val if normalize else None
        }, output_dir / f'spec_{idx:06d}.pt')
    
    # Save normalization statistics
    if normalize:
        stats = {
            'normalized': True,
            'method': 'joint_max',
            'global_max': np.max(all_max_values),
            'global_mean_max': np.mean(all_max_values),
            'global_std_max': np.std(all_max_values),
            'percentile_99': np.percentile(all_max_values, 99)
        }
        torch.save(stats, output_dir / 'normalization_stats.pt')
        
        print(f"✅ Normalization stats:")
        print(f"   Global max: {stats['global_max']:.2f}")
        print(f"   Mean max: {stats['global_mean_max']:.2f}")
        print(f"   99th percentile: {stats['percentile_99']:.2f}")
    
    print(f"✅ Saved {len(reverb_files)} normalized spectrograms to {output_dir}")

if __name__ == '__main__':
    precompute_all_spectrograms(
        data_dir='/scratch/egbueze.m/reverb_dataset',
        output_dir='/scratch/egbueze.m/precomputed_specs_normalized',
        normalize=True
    )