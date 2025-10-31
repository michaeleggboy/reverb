import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
from audio_utils import audio_to_spectrogram, resize_spectrogram


class DereverbDataset(Dataset):
    def __init__(self, reverb_dir, clean_dir):
        """
        Args:
            reverb_dir: path to folder with reverberant audio files
            clean_dir: path to folder with clean audio files
        """
        reverb_path = Path(reverb_dir)
        clean_path = Path(clean_dir)
        
        self.reverb_files = sorted(list(reverb_path.glob('*.flac')))
        self.clean_files = sorted(list(clean_path.glob('*.flac')))
        
        assert len(self.reverb_files) == len(self.clean_files), \
            f"Mismatch: {len(self.reverb_files)} reverb vs {len(self.clean_files)} clean files"
        
        print(f"Loaded dataset with {len(self.reverb_files)} pairs")
    
    def __len__(self):
        return len(self.reverb_files)
    
    def __getitem__(self, idx):
        # Load audio
        reverb_audio, sr = torchaudio.load(str(self.reverb_files[idx]))
        clean_audio, _ = torchaudio.load(str(self.clean_files[idx]))
        
        # Compute spectrograms (magnitude only, phase not needed for training)
        reverb_spec = self.compute_spectrogram(reverb_audio)
        clean_spec = self.compute_spectrogram(clean_audio)
        
        return reverb_spec, clean_spec
    
    def compute_spectrogram(self, audio):
        """
        Convert audio to magnitude spectrogram resized to 256x256
        Uses audio_utils for consistency with inference
        """
        # Convert to spectrogram (returns magnitude and phase)
        magnitude, _ = audio_to_spectrogram(audio, n_fft=512, hop_length=256)
        
        # Resize to 256x256 for model input
        magnitude_resized = resize_spectrogram(magnitude, target_size=(256, 256))
        
        # Remove batch dimension added by resize_spectrogram
        magnitude_resized = magnitude_resized.squeeze(0)
        
        return magnitude_resized
