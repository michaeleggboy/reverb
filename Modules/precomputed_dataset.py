import torch
from torch.utils.data import Dataset
from pathlib import Path


class PrecomputedDataset(Dataset):
    def __init__(self, spec_dir, return_metadata=False):
        self.spec_dir = Path(spec_dir)
        self.spec_files = sorted(self.spec_dir.glob('spec_*.pt'))
        self.return_metadata = return_metadata
        
        # Load preprocessing stats
        stats_file = self.spec_dir / 'preprocessing_stats.pt'
        if stats_file.exists():
            self.stats = torch.load(stats_file, weights_only=False)
            print(f"Found {len(self.spec_files)} pre-computed spectrograms")
            print(f"  Version: {self.stats.get('version', 1)}")
            print(f"  Mean diff: {self.stats.get('mean_difference', 'N/A'):.4f}")
        else:
            self.stats = None
            print(f"Found {len(self.spec_files)} pre-computed spectrograms")
    
    def __len__(self):
        return len(self.spec_files)
    
    def __getitem__(self, idx):
        data = torch.load(self.spec_files[idx], weights_only=False)
        
        reverb = data['reverb']
        clean = data['clean']
        
        if reverb.dim() == 2:
            reverb = reverb.unsqueeze(0)
        if clean.dim() == 2:
            clean = clean.unsqueeze(0)
        
        if self.return_metadata:
            return {
                'reverb': reverb,
                'clean': clean,
                'reverb_phase': data.get('reverb_phase'),
                'clean_phase': data.get('clean_phase'),
                'original_shape': data.get('original_shape'),
                'reverb_file': data.get('reverb_file'),
            }
        
        return reverb, clean