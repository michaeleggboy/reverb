import unittest
import torch
import tempfile
import soundfile as sf
import numpy as np
from pathlib import Path

from dataset import DereverbDataset


class TestDereverbDataset(unittest.TestCase):
    """Test DereverbDataset class"""
    
    @classmethod
    def setUpClass(cls):
        """Create temporary dataset for testing"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.reverb_dir = Path(cls.temp_dir) / 'reverb'
        cls.clean_dir = Path(cls.temp_dir) / 'clean'
        cls.reverb_dir.mkdir()
        cls.clean_dir.mkdir()
        
        # Create 10 dummy audio files
        sr = 16000
        duration = 1.0
        num_files = 10
        
        for i in range(num_files):
            # Generate random audio
            audio = np.random.randn(int(sr * duration)).astype(np.float32) * 0.5
            
            # Save reverb version
            sf.write(
                cls.reverb_dir / f'sample_{i:03d}.flac',
                audio,
                sr
            )
            
            # Save clean version (slightly different)
            clean_audio = audio * 0.8  # Simulate cleaning
            sf.write(
                cls.clean_dir / f'sample_{i:03d}.flac',
                clean_audio,
                sr
            )
    
    def test_dataset_length(self):
        """Test __len__ returns correct count"""
        dataset = DereverbDataset(self.reverb_dir, self.clean_dir)
        self.assertEqual(len(dataset), 10)
    
    def test_getitem_returns_correct_types(self):
        """Test __getitem__ returns tensors"""
        dataset = DereverbDataset(self.reverb_dir, self.clean_dir)
        reverb_spec, clean_spec = dataset[0]
        
        # Check types
        self.assertIsInstance(reverb_spec, torch.Tensor)
        self.assertIsInstance(clean_spec, torch.Tensor)
    
    def test_getitem_returns_correct_shapes(self):
        """Test output shapes are 256x256"""
        dataset = DereverbDataset(self.reverb_dir, self.clean_dir)
        reverb_spec, clean_spec = dataset[0]
        
        # Both should be [1, 256, 256]
        self.assertEqual(reverb_spec.shape, torch.Size([1, 256, 256]))
        self.assertEqual(clean_spec.shape, torch.Size([1, 256, 256]))
    
    def test_all_indices_accessible(self):
        """Test all dataset indices can be accessed"""
        dataset = DereverbDataset(self.reverb_dir, self.clean_dir)
        
        for i in range(len(dataset)):
            reverb_spec, clean_spec = dataset[i]
            self.assertIsNotNone(reverb_spec)
            self.assertIsNotNone(clean_spec)
    
    def test_mismatch_raises_error(self):
        """Test that file count mismatch raises error"""
        # Create mismatched directories
        temp_dir = tempfile.mkdtemp()
        reverb = Path(temp_dir) / 'reverb'
        clean = Path(temp_dir) / 'clean'
        reverb.mkdir()
        clean.mkdir()
        
        # Create 5 reverb files
        sr = 16000
        for i in range(5):
            audio = np.random.randn(sr).astype(np.float32)
            sf.write(reverb / f'sample_{i}.flac', audio, sr)
        
        # Create only 3 clean files (mismatch!)
        for i in range(3):
            audio = np.random.randn(sr).astype(np.float32)
            sf.write(clean / f'sample_{i}.flac', audio, sr)
        
        # Should raise assertion error
        with self.assertRaises(AssertionError):
            dataset = DereverbDataset(reverb, clean)
    
    def test_spectrogram_values(self):
        """Test spectrogram values are valid"""
        dataset = DereverbDataset(self.reverb_dir, self.clean_dir)
        reverb_spec, clean_spec = dataset[0]
        
        # Check for invalid values
        self.assertFalse(torch.isnan(reverb_spec).any())
        self.assertFalse(torch.isinf(reverb_spec).any())
        self.assertFalse(torch.isnan(clean_spec).any())
        self.assertFalse(torch.isinf(clean_spec).any())
        
        # Magnitude should be non-negative
        self.assertTrue(torch.all(reverb_spec >= 0))
        self.assertTrue(torch.all(clean_spec >= 0))


if __name__ == '__main__':
    unittest.main(verbosity=2)
