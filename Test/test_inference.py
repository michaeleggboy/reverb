import unittest
import torch
import torchaudio
import tempfile
import soundfile as sf
import numpy as np
from pathlib import Path

from inference import dereverb_audio, dereverb_batch
from unet import UNet


class TestInference(unittest.TestCase):
    """Test inference functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Create test files and model"""
        cls.temp_dir = tempfile.mkdtemp()
        
        # Create dummy audio file
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
        
        cls.test_audio_path = Path(cls.temp_dir) / 'test_reverb.flac'
        sf.write(cls.test_audio_path, audio, sr)
        
        # Create dummy model checkpoint
        model = UNet(in_channels=1, out_channels=1)
        cls.checkpoint_path = Path(cls.temp_dir) / 'test_model.pth'
        torch.save({
            'epoch': 10,
            'model_state_dict': model.state_dict(),
            'val_loss': 0.35
        }, cls.checkpoint_path)
    
    def test_dereverb_creates_output(self):
        """Test that dereverb creates output file"""
        output_path = Path(self.temp_dir) / 'output_clean.wav'
        
        clean_audio, sr = dereverb_audio(
            input_audio_path=self.test_audio_path,
            output_audio_path=output_path,
            model_path=self.checkpoint_path,
            device='cpu'
        )
        
        # Check output file exists
        self.assertTrue(output_path.exists())
        
        # Check output is valid
        self.assertIsInstance(clean_audio, torch.Tensor)
        self.assertGreater(sr, 0)
    
    def test_output_audio_valid_range(self):
        """Test output audio is in valid range"""
        output_path = Path(self.temp_dir) / 'output_test.wav'
        
        clean_audio, sr = dereverb_audio(
            self.test_audio_path,
            output_path,
            self.checkpoint_path,
            device='cpu'
        )
        
        # Audio should be in [-1, 1] range
        self.assertGreaterEqual(clean_audio.min().item(), -1.0)
        self.assertLessEqual(clean_audio.max().item(), 1.0)
    
    def test_output_matches_input_length(self):
        """Test output has same duration as input"""
        output_path = Path(self.temp_dir) / 'output_length.wav'
        
        # Load input
        input_audio, input_sr = torchaudio.load(str(self.test_audio_path))
        input_length = input_audio.shape[-1]
        
        # Process
        clean_audio, sr = dereverb_audio(
            self.test_audio_path,
            output_path,
            self.checkpoint_path,
            device='cpu'
        )
        
        # Check length matches (within small tolerance)
        output_length = clean_audio.shape[-1] if clean_audio.dim() == 1 else clean_audio.shape[1]
        self.assertAlmostEqual(output_length, input_length, delta=512)
    
    def test_handles_stereo_input(self):
        """Test that stereo input is handled"""
        # Create stereo file
        sr = 16000
        stereo_audio = np.random.randn(2, sr).astype(np.float32) * 0.5
        stereo_path = Path(self.temp_dir) / 'stereo.flac'
        sf.write(stereo_path, stereo_audio.T, sr)
        
        output_path = Path(self.temp_dir) / 'stereo_clean.wav'
        
        # Should not crash
        clean_audio, sr = dereverb_audio(
            stereo_path,
            output_path,
            self.checkpoint_path,
            device='cpu'
        )
        
        # Should produce mono output
        self.assertIsNotNone(clean_audio)
    
    def test_output_no_nan_inf(self):
        """Test output doesn't contain NaN or Inf"""
        output_path = Path(self.temp_dir) / 'output_valid.wav'
        
        clean_audio, sr = dereverb_audio(
            self.test_audio_path,
            output_path,
            self.checkpoint_path,
            device='cpu'
        )
        
        self.assertFalse(torch.isnan(clean_audio).any())
        self.assertFalse(torch.isinf(clean_audio).any())


class TestBatchInference(unittest.TestCase):
    """Test batch processing"""
    
    @classmethod
    def setUpClass(cls):
        """Create test files"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.input_dir = Path(cls.temp_dir) / 'input'
        cls.output_dir = Path(cls.temp_dir) / 'output'
        cls.input_dir.mkdir()
        
        # Create 5 test files
        sr = 16000
        for i in range(5):
            audio = (np.random.randn(sr) * 0.5).astype(np.float32)
            sf.write(cls.input_dir / f'test_{i}.wav', audio, sr)
        
        # Create dummy model
        model = UNet(in_channels=1, out_channels=1)
        cls.checkpoint_path = Path(cls.temp_dir) / 'model.pth'
        torch.save(model.state_dict(), cls.checkpoint_path)
    
    def test_batch_processing(self):
        """Test processing multiple files"""
        dereverb_batch(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            model_path=self.checkpoint_path,
            device='cpu'
        )
        
        # Check output files created
        output_files = list(self.output_dir.glob('*.wav'))
        self.assertEqual(len(output_files), 5)
    
    def test_batch_output_naming(self):
        """Test batch output files are named correctly"""
        dereverb_batch(
            self.input_dir,
            self.output_dir,
            self.checkpoint_path,
            device='cpu'
        )
        
        # Check naming convention (clean_*)
        for output_file in self.output_dir.glob('*.wav'):
            self.assertTrue(output_file.name.startswith('clean_'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
