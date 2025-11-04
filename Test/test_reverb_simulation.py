import unittest
import numpy as np
import pyroomacoustics as pra


class TestReverbGeneration(unittest.TestCase):
    """Test pyroomacoustics reverb generation"""
    
    def test_room_creation(self):
        """Test creating a room"""
        room = pra.ShoeBox(
            [5, 4, 3],
            fs=16000,
            materials=pra.Material(0.5),
            max_order=15
        )
        
        self.assertIsNotNone(room)
    
    def test_reverb_simulation(self):
        """Test simulating reverb"""
        # Create test audio
        sr = 16000
        audio = np.random.randn(sr).astype(np.float32) * 0.3
        
        # Create room
        room = pra.ShoeBox([5, 4, 3], fs=sr, materials=pra.Material(0.5), max_order=10)
        
        # Add source and mic
        room.add_source([2, 2, 1.5], signal=audio)
        room.add_microphone([3, 2, 1.2])
        
        # Simulate
        room.simulate()
        reverb_audio = room.mic_array.signals[0, :]
        
        # Check output
        self.assertIsInstance(reverb_audio, np.ndarray)
        self.assertGreater(len(reverb_audio), 0)
    
    def test_reverb_longer_than_source(self):
        """Test that reverb is longer due to tail"""
        sr = 16000
        audio = np.random.randn(sr).astype(np.float32) * 0.3
        
        room = pra.ShoeBox([8, 6, 3], fs=sr, materials=pra.Material(0.8), max_order=15)
        room.add_source([2, 2, 1.5], signal=audio)
        room.add_microphone([6, 4, 1.2])
        room.simulate()
        
        reverb_audio = room.mic_array.signals[0, :]
        
        # Reverb should be longer (has tail)
        self.assertGreaterEqual(len(reverb_audio), len(audio))
    
    def test_different_rt60_values(self):
        """Test various RT60 values work"""
        sr = 16000
        audio = np.random.randn(sr * 0.5).astype(np.float32) * 0.3
        
        for rt60 in [0.2, 0.5, 1.0]:
            room = pra.ShoeBox(
                [5, 4, 3],
                fs=sr,
                materials=pra.Material(rt60),
                max_order=15
            )
            room.add_source([2, 2, 1.5], signal=audio)
            room.add_microphone([3, 2, 1.2])
            room.simulate()
            
            reverb_audio = room.mic_array.signals[0, :]
            
            # Should produce valid audio
            self.assertFalse(np.isnan(reverb_audio).any())
            self.assertFalse(np.isinf(reverb_audio).any())
            self.assertGreater(len(reverb_audio), 0)
    
    def test_normalization_prevents_clipping(self):
        """Test that normalization keeps audio in valid range"""
        reverb_audio = np.array([2.5, -1.8, 0.5, 3.2, -2.1])
        
        # Normalize
        max_val = np.max(np.abs(reverb_audio))
        normalized = reverb_audio / max_val * 0.9
        
        # Check range
        self.assertGreaterEqual(normalized.min(), -1.0)
        self.assertLessEqual(normalized.max(), 1.0)
        self.assertAlmostEqual(np.max(np.abs(normalized)), 0.9, places=5)


if __name__ == '__main__':
    unittest.main(verbosity=2)

