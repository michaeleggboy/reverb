import unittest
import torch
import tempfile
import soundfile as sf
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, random_split

from dataset import DereverbDataset
from unet import UNet
from inference import dereverb_audio


class TestEndToEnd(unittest.TestCase):
    """Test complete pipeline from data to inference"""
    
    @classmethod
    def setUpClass(cls):
        """Create mini dataset and train mini model"""
        cls.temp_dir = tempfile.mkdtemp()
        
        # Create small dataset (20 samples)
        reverb_dir = Path(cls.temp_dir) / 'reverb'
        clean_dir = Path(cls.temp_dir) / 'clean'
        reverb_dir.mkdir(parents=True)
        clean_dir.mkdir(parents=True)
        
        sr = 16000
        for i in range(20):
            # Reverberant audio
            reverb = np.random.randn(sr).astype(np.float32) * 0.5
            sf.write(reverb_dir / f'sample_{i:03d}.flac', reverb, sr)
            
            # Clean audio (similar but less energy)
            clean = reverb * 0.7
            sf.write(clean_dir / f'sample_{i:03d}.flac', clean, sr)
        
        cls.dataset = DereverbDataset(reverb_dir, clean_dir)
        cls.checkpoint_dir = Path(cls.temp_dir) / 'checkpoints'
        cls.checkpoint_dir.mkdir()
    
    def test_dataloader_works(self):
        """Test DataLoader with dataset"""
        dataloader = DataLoader(self.dataset, batch_size=4, shuffle=True)
        
        # Get one batch
        reverb_batch, clean_batch = next(iter(dataloader))
        
        self.assertEqual(reverb_batch.shape, torch.Size([4, 1, 256, 256]))
        self.assertEqual(clean_batch.shape, torch.Size([4, 1, 256, 256]))
    
    def test_mini_training(self):
        """Test that model can train for a few iterations"""
        # Create mini train/val split
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_dataset, val_dataset = random_split(
            self.dataset,
            [train_size, val_size]
        )
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        
        # Initialize model
        model = UNet(in_channels=1, out_channels=1)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Train for 2 iterations
        model.train()
        losses = []
        
        for i, (reverb, clean) in enumerate(train_loader):
            if i >= 2:  # Only 2 iterations
                break
            
            pred = model(reverb)
            loss = criterion(pred, clean)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        # Check that loss is reasonable
        self.assertLess(losses[0], 10.0)  # Not exploding
        self.assertGreater(losses[0], 0.0)  # Not zero
    
    def test_save_and_load_checkpoint(self):
        """Test saving and loading model checkpoint"""
        model = UNet(in_channels=1, out_channels=1)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / 'test_checkpoint.pth'
        torch.save({
            'epoch': 5,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': 0.5,
            'val_loss': 0.6
        }, checkpoint_path)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        
        # Verify contents
        self.assertEqual(checkpoint['epoch'], 5)
        self.assertIn('model_state_dict', checkpoint)
        self.assertIn('optimizer_state_dict', checkpoint)
        
        # Load into new model
        new_model = UNet(in_channels=1, out_channels=1)
        new_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Parameters should match
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))
    
    def test_inference_after_training(self):
        """Test full pipeline: train -> save -> load -> inference"""
        # Train mini model
        model = UNet(in_channels=1, out_channels=1)
        train_loader = DataLoader(self.dataset, batch_size=4)
        
        model.train()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Train 1 epoch
        for reverb, clean in train_loader:
            pred = model(reverb)
            loss = criterion(pred, clean)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Save model
        checkpoint_path = self.checkpoint_dir / 'trained_model.pth'
        torch.save(model.state_dict(), checkpoint_path)
        
        # Create test audio
        test_audio_path = Path(self.temp_dir) / 'test_input.wav'
        sr = 16000
        audio = (np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)) * 0.5).astype(np.float32)
        sf.write(test_audio_path, audio, sr)
        
        # Run inference
        output_path = Path(self.temp_dir) / 'test_output.wav'
        clean_audio, sr = dereverb_audio(
            test_audio_path,
            output_path,
            checkpoint_path,
            device='cpu'
        )
        
        # Verify output
        self.assertTrue(output_path.exists())
        self.assertIsNotNone(clean_audio)
        self.assertFalse(torch.isnan(clean_audio).any())


if __name__ == '__main__':
    unittest.main(verbosity=2)

