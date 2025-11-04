import unittest
import torch
import torchaudio
import numpy as np
from pathlib import Path

from audio_utils import (
    audio_to_spectrogram,
    spectrogram_to_audio,
    resize_spectrogram,
    unresize_spectrogram
)


class TestAudioToSpectrogram(unittest.TestCase):
    """Test audio to spectrogram conversion"""
    
    def setUp(self):
        """Create test audio"""
        # Create 1 second of sine wave at 440 Hz
        sr = 16000
        duration = 1.0
        t = torch.linspace(0, duration, int(sr * duration))
        self.audio = torch.sin(2 * np.pi * 440 * t).unsqueeze(0)  # [1, 16000]
        self.sr = sr
    
    def test_output_shapes(self):
        """Test that output shapes are correct"""
        magnitude, phase = audio_to_spectrogram(self.audio)
        
        # Check shapes
        self.assertEqual(magnitude.dim(), 3)  # [channels, freq, time]
        self.assertEqual(phase.dim(), 3)
        self.assertEqual(magnitude.shape, phase.shape)
        
        # Check frequency bins (n_fft=512 -> 257 bins)
        self.assertEqual(magnitude.shape[1], 257)
    
    def test_magnitude_non_negative(self):
        """Test that magnitude is non-negative"""
        magnitude, _ = audio_to_spectrogram(self.audio)
        
        self.assertTrue(torch.all(magnitude >= 0))
        self.assertFalse(torch.isnan(magnitude).any())
        self.assertFalse(torch.isinf(magnitude).any())
    
    def test_phase_range(self):
        """Test that phase is in valid range [-π, π]"""
        _, phase = audio_to_spectrogram(self.audio)
        
        self.assertTrue(torch.all(phase >= -np.pi))
        self.assertTrue(torch.all(phase <= np.pi))
        self.assertFalse(torch.isnan(phase).any())


class TestSpectrogramToAudio(unittest.TestCase):
    """Test spectrogram to audio conversion"""
    
    def setUp(self):
        """Create test spectrogram"""
        sr = 16000
        t = torch.linspace(0, 1.0, sr)
        self.audio = torch.sin(2 * np.pi * 440 * t).unsqueeze(0)
        self.magnitude, self.phase = audio_to_spectrogram(self.audio)
    
    def test_reconstruction(self):
        """Test that audio can be reconstructed"""
        reconstructed = spectrogram_to_audio(self.magnitude, self.phase)
        
        # Check shape
        self.assertEqual(reconstructed.dim(), 1)
        
        # Check it's close to original (won't be exact due to windowing)
        # Allow 5% error
        self.assertLess(
            torch.mean(torch.abs(reconstructed - self.audio.squeeze())).item(),
            0.05
        )
    
    def test_shape_mismatch_handling(self):
        """Test that shape mismatches are handled"""
        # Create mismatched shapes
        magnitude_wrong = self.magnitude[:, :256, :186]  # Smaller
        phase_correct = self.phase  # Original size
        
        # Should NOT crash - should handle mismatch
        audio = spectrogram_to_audio(magnitude_wrong, phase_correct)
        
        # Should produce valid audio
        self.assertFalse(torch.isnan(audio).any())
        self.assertFalse(torch.isinf(audio).any())
        self.assertTrue(audio.shape[0] > 0)
    
    def test_handles_nan_in_magnitude(self):
        """Test that NaN values are handled"""
        magnitude_with_nan = self.magnitude.clone()
        magnitude_with_nan[0, 10, 10] = float('nan')
        
        # Should not crash
        audio = spectrogram_to_audio(magnitude_with_nan, self.phase)
        
        # Output should be valid
        self.assertFalse(torch.isnan(audio).any())
    
    def test_handles_inf_in_magnitude(self):
        """Test that Inf values are handled"""
        magnitude_with_inf = self.magnitude.clone()
        magnitude_with_inf[0, 10, 10] = float('inf')
        
        # Should not crash
        audio = spectrogram_to_audio(magnitude_with_inf, self.phase)
        
        # Output should be valid
        self.assertFalse(torch.isinf(audio).any())


class TestResizeSpectrogram(unittest.TestCase):
    """Test spectrogram resizing"""
    
    def setUp(self):
        """Create test spectrogram"""
        self.spec = torch.randn(1, 257, 187)  # [C, H, W]
    
    def test_resize_to_256x256(self):
        """Test resizing to 256x256"""
        resized = resize_spectrogram(self.spec, target_size=(256, 256))
        
        # Check output shape
        self.assertEqual(resized.shape, torch.Size([1, 1, 256, 256]))
    
    def test_resize_2d_input(self):
        """Test with 2D input"""
        spec_2d = torch.randn(257, 187)
        resized = resize_spectrogram(spec_2d, target_size=(256, 256))
        
        self.assertEqual(resized.shape[2:], torch.Size([256, 256]))
    
    def test_resize_reversible(self):
        """Test that resize and unresize are approximately reversible"""
        original_size = self.spec.shape[-2:]
        
        # Resize to 256x256 and back
        resized = resize_spectrogram(self.spec, (256, 256))
        unresized = unresize_spectrogram(resized, original_size)
        unresized = unresized.squeeze(0)  # Remove batch dim
        
        # Should be close to original (not exact due to interpolation)
        diff = torch.mean(torch.abs(unresized - self.spec))
        self.assertLess(diff.item(), 0.1)  # Allow small interpolation error


class TestRoundTrip(unittest.TestCase):
    """Test complete round-trip: audio -> spec -> audio"""
    
    def test_sine_wave_roundtrip(self):
        """Test reconstruction of sine wave"""
        # Create pure sine wave
        sr = 16000
        freq = 440
        duration = 1.0
        t = torch.linspace(0, duration, int(sr * duration))
        original = torch.sin(2 * np.pi * freq * t).unsqueeze(0)
        
        # Convert to spectrogram
        magnitude, phase = audio_to_spectrogram(original)
        
        # Convert back
        reconstructed = spectrogram_to_audio(magnitude, phase)
        
        # Trim to same length
        min_len = min(original.shape[-1], reconstructed.shape[-1])
        original = original[..., :min_len]
        reconstructed = reconstructed[:min_len]
        
        # Check similarity (STFT is not perfectly invertible)
        correlation = torch.corrcoef(torch.stack([
            original.squeeze(),
            reconstructed
        ]))[0, 1]
        
        # Should be highly correlated (>0.95)
        self.assertGreater(correlation.item(), 0.95)
    
    def test_speech_roundtrip(self):
        """Test with real speech if available"""
        # Try to load test file
        test_file = Path('test_data/sample.flac')
        
        if not test_file.exists():
            self.skipTest("No test audio file available")
        
        original, sr = torchaudio.load(str(test_file))
        
        # Convert to mono
        if original.shape[0] > 1:
            original = original.mean(dim=0, keepdim=True)
        
        # Round trip
        magnitude, phase = audio_to_spectrogram(original)
        reconstructed = spectrogram_to_audio(magnitude, phase)
        
        # Should be similar
        min_len = min(original.shape[-1], reconstructed.shape[-1])
        original = original[..., :min_len]
        reconstructed = reconstructed[:min_len].unsqueeze(0)
        
        # SNR should be high
        noise = original - reconstructed
        snr = 10 * torch.log10(
            torch.mean(original ** 2) / torch.mean(noise ** 2)
        )
        
        # SNR should be > 30 dB for good reconstruction
        self.assertGreater(snr.item(), 30.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)

