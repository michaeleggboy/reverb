import os
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset

class DereverbDataset(Dataset):
    def __init__(self, reverb_dir, clean_dir):
        """
        Args:
            reverb_dir: path to folder with reverberant audio files
            clean_dir: path to folder with clean audio files
        """
        self.reverb_files = sorted([os.path.join(reverb_dir, f) 
                                    for f in os.listdir(reverb_dir) 
                                    if f.endswith('.wav')])
        self.clean_files = sorted([os.path.join(clean_dir, f) 
                                   for f in os.listdir(clean_dir) 
                                   if f.endswith('.wav')])
        
        assert len(self.reverb_files) == len(self.clean_files), \
            "Mismatch between reverb and clean files"
    
    def __len__(self):
        return len(self.reverb_files)
    
    def __getitem__(self, idx):
        # Load audio
        reverb_audio, sr = torchaudio.load(self.reverb_files[idx])
        clean_audio, _ = torchaudio.load(self.clean_files[idx])
        
        # Compute spectrograms
        reverb_spec = self.compute_spectrogram(reverb_audio)
        clean_spec = self.compute_spectrogram(clean_audio)
        
        return reverb_spec, clean_spec
    
    def compute_spectrogram(self, audio):
        # STFT parameters
        n_fft = 512
        hop_length = 256
        
        # Compute STFT
        spec = torch.stft(
            audio, 
            n_fft=n_fft, 
            hop_length=hop_length,
            return_complex=True
        )
        
        # Get magnitude
        mag_spec = torch.abs(spec)
        
        # Resize to 256x256
        mag_spec = F.interpolate(mag_spec.unsqueeze(0), 
                                 size=(256, 256), 
                                 mode='bilinear').squeeze(0)
        
        return mag_spec
