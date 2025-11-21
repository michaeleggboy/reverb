import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
from audio_utils import audio_to_spectrogram, resize_spectrogram


def precompute_all_spectrograms():
    reverb_dir = Path('/scratch/egbueze.m/reverb_dataset/reverb')
    clean_dir = Path('/scratch/egbueze.m/reverb_dataset/clean')
    output_dir = Path('/scratch/egbueze.m/precomputed_specs')
    output_dir.mkdir(exist_ok=True)
    
    reverb_files = sorted(list(reverb_dir.glob('*.flac')))
    clean_files = sorted(list(clean_dir.glob('*.flac')))
    
    print(f"Pre-computing {len(reverb_files)} spectrograms...")
    
    for idx, (reverb_file, clean_file) in enumerate(tqdm(zip(reverb_files, clean_files))):
        reverb_audio, _ = torchaudio.load(str(reverb_file))
        clean_audio, _ = torchaudio.load(str(clean_file))
        
        reverb_mag, _ = audio_to_spectrogram(reverb_audio, n_fft=512, hop_length=256)
        clean_mag, _ = audio_to_spectrogram(clean_audio, n_fft=512, hop_length=256)
        
        reverb_spec = resize_spectrogram(reverb_mag, target_size=(256, 256)).squeeze(0)
        clean_spec = resize_spectrogram(clean_mag, target_size=(256, 256)).squeeze(0)
        
        torch.save({
            'reverb': reverb_spec,
            'clean': clean_spec
        }, output_dir / f'spec_{idx:06d}.pt')
    
    print(f"Saved to {output_dir}")

if __name__ == '__main__':
    precompute_all_spectrograms()