import pyroomacoustics as pra
import numpy as np
import soundfile as sf
import os
from pathlib import Path

def create_reverb_dataset(clean_audio_dir, output_dir, num_samples=100):
    """
    Generate reverberant versions of clean audio files
    
    Args:
        clean_audio_dir: folder with clean .flac files
        output_dir: where to save reverb/clean pairs
        num_samples: number of different room configs per audio file
    """
    
    # Create output directories
    reverb_dir = Path(output_dir) / 'reverb'
    clean_dir = Path(output_dir) / 'clean'
    reverb_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all clean audio files (FLAC)
    clean_files = list(Path(clean_audio_dir).glob('*.flac'))
    
    sample_idx = 0
    for clean_file in clean_files:
        audio, sr = sf.read(clean_file)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Generate multiple room configs for each audio
        for i in range(num_samples):
            # Random room parameters
            room_dim = np.random.uniform([3, 3, 2.5], [10, 10, 4])
            rt60 = np.random.uniform(0.2, 1.0)
            
            # Create room
            room = pra.ShoeBox(
                room_dim,
                fs=sr,
                materials=pra.Material(rt60),
                max_order=15
            )
            
            # Random source and mic positions
            source_pos = np.random.uniform(
                [0.5, 0.5, 0.5], 
                room_dim - [0.5, 0.5, 0.5]
            )
            mic_pos = np.random.uniform(
                [0.5, 0.5, 0.5],
                room_dim - [0.5, 0.5, 0.5]
            )
            
            # Add source and microphone
            room.add_source(source_pos, signal=audio)
            room.add_microphone(mic_pos)
            
            # Simulate
            room.simulate()
            reverb_audio = room.mic_array.signals[0, :]
            
            # Normalize
            reverb_audio = reverb_audio / np.max(np.abs(reverb_audio)) * 0.9
            clean_audio_normalized = audio / np.max(np.abs(audio)) * 0.9
            
            # Match lengths
            min_len = min(len(reverb_audio), len(clean_audio_normalized))
            reverb_audio = reverb_audio[:min_len]
            clean_audio_normalized = clean_audio_normalized[:min_len]
            
            # Save as FLAC (lossless, smaller than WAV)
            reverb_path = reverb_dir / f'sample_{sample_idx:05d}.flac'
            clean_path = clean_dir / f'sample_{sample_idx:05d}.flac'
            
            sf.write(reverb_path, reverb_audio, sr)
            sf.write(clean_path, clean_audio_normalized, sr)
            
            print(f"Generated {sample_idx}: RT60={rt60:.2f}s, Room={room_dim}")
            sample_idx += 1

# Usage
create_reverb_dataset(
    clean_audio_dir='./speech',  # Folder with .flac files
    output_dir='./data',
    num_samples=5
)
