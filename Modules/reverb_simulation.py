import pyroomacoustics as pra
import numpy as np
import soundfile as sf
from pathlib import Path

def create_reverb_from_librispeech(
    librispeech_root,
    output_dir,
    subset='train-clean-100',
    rooms_per_audio=3
):
    """
    Generate reverberant versions from LibriSpeech clean data
    
    Args:
        librispeech_root: Path to LibriSpeech root (contains train-clean-100/, etc.)
        output_dir: Where to save reverb/clean pairs
        subset: Which LibriSpeech subset to use ('train-clean-100', 'dev-clean', etc.)
        rooms_per_audio: Number of different room configs per audio file
    """
    
    # Setup paths
    subset_path = Path(librispeech_root) / subset
    reverb_dir = Path(output_dir) / 'reverb'
    clean_dir = Path(output_dir) / 'clean'
    reverb_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    
    # LibriSpeech structure: subset/speaker_id/chapter_id/*.flac
    sample_idx = 0
    
    # Iterate through all speakers
    for speaker_dir in sorted(subset_path.iterdir()):
        if not speaker_dir.is_dir():
            continue
            
        speaker_id = speaker_dir.name
        print(f"Processing speaker {speaker_id}...")
        
        # Iterate through all chapters for this speaker
        for chapter_dir in sorted(speaker_dir.iterdir()):
            if not chapter_dir.is_dir():
                continue
                
            chapter_id = chapter_dir.name
            
            # Get all FLAC files in this chapter
            flac_files = sorted(chapter_dir.glob('*.flac'))
            
            for flac_file in flac_files:
                # Load clean audio
                audio, sr = sf.read(flac_file)
                
                # Convert to mono if stereo
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                
                # Generate multiple reverberant versions
                for room_idx in range(rooms_per_audio):
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
                    
                    # Random positions
                    source_pos = np.random.uniform(
                        [0.5, 0.5, 0.5],
                        room_dim - [0.5, 0.5, 0.5]
                    )
                    mic_pos = np.random.uniform(
                        [0.5, 0.5, 0.5],
                        room_dim - [0.5, 0.5, 0.5]
                    )
                    
                    # Simulate
                    room.add_source(source_pos, signal=audio)
                    room.add_microphone(mic_pos)
                    room.simulate()
                    reverb_audio = room.mic_array.signals[0, :]
                    
                    # Normalize
                    reverb_audio = reverb_audio / (np.max(np.abs(reverb_audio)) + 1e-8) * 0.9
                    clean_audio_norm = audio / (np.max(np.abs(audio)) + 1e-8) * 0.9
                    
                    # Match lengths
                    min_len = min(len(reverb_audio), len(clean_audio_norm))
                    reverb_audio = reverb_audio[:min_len]
                    clean_audio_norm = clean_audio_norm[:min_len]
                    
                    # Save with descriptive names
                    reverb_path = reverb_dir / f'{speaker_id}_{chapter_id}_{flac_file.stem}_room{room_idx}.flac'
                    clean_path = clean_dir / f'{speaker_id}_{chapter_id}_{flac_file.stem}_room{room_idx}.flac'
                    
                    sf.write(reverb_path, reverb_audio, sr)
                    sf.write(clean_path, clean_audio_norm, sr)
                    
                    if sample_idx % 100 == 0:
                        print(f"  Generated {sample_idx} samples (RT60={rt60:.2f}s)")
                    
                    sample_idx += 1
    
    print(f"\nTotal samples generated: {sample_idx}")
    print(f"Reverb audio: {reverb_dir}")
    print(f"Clean audio: {clean_dir}")

# Usage
create_reverb_from_librispeech(
    librispeech_root='./LibriSpeech',
    output_dir='./dereverb_dataset',
    subset='train-clean-100',  # ~100 hours
    rooms_per_audio=3  # 3 different rooms per utterance
)
