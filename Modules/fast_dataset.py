import torch
from torch.utils.data import Dataset
from pathlib import Path


class PrecomputedDataset(Dataset):
    def __init__(self, spec_dir):
        self.spec_files = sorted(Path(spec_dir).glob('*.pt'))
        print(f"Found {len(self.spec_files)} pre-computed spectrograms")
    
    def __len__(self):
        return len(self.spec_files)
    
    def __getitem__(self, idx):
        data = torch.load(self.spec_files[idx], weights_only=False)
        return data['reverb'], data['clean']