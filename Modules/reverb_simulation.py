import pyroomacoustics as pra
import numpy as np
import soundfile as sf
from pathlib import Path
import json
import time


def create_reverb_from_librispeech(
    librispeech_root,
    output_dir,
    subset='train-clean-100',
    rooms_per_audio=3,
    checkpoint_file=None
):
    """
    Generate reverberant versions from LibriSpeech clean data
    With checkpointing to resume if interrupted

    Args:
        librispeech_root: Path to LibriSpeech root (contains train-clean-100/, etc.)
        output_dir: Where to save reverb/clean pairs
        subset: Which LibriSpeech subset to use ('train-clean-100', 'dev-clean', etc.)
        rooms_per_audio: Number of different room configs per audio file
        checkpoint_file: Path to save/load progress (default: output_dir/checkpoint.json)
    """

    # Setup paths
    subset_path = Path(librispeech_root) / subset
    reverb_dir = Path(output_dir) / 'reverb'
    clean_dir = Path(output_dir) / 'clean'
    reverb_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    
    # Checkpoint file
    if checkpoint_file is None:
        checkpoint_file = Path(output_dir) / 'generation_checkpoint.json'
    else:
        checkpoint_file = Path(checkpoint_file)
    
    # Load checkpoint if exists
    processed_files = set()
    sample_idx = 0
    
    if checkpoint_file.exists():
        print(f"📂 Found checkpoint: {checkpoint_file}")
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
            processed_files = set(checkpoint_data['processed_files'])
            sample_idx = checkpoint_data['sample_count']
        print(f"✓ Resuming from {sample_idx} samples")
        print(f"✓ Already processed {len(processed_files)} source files")
    else:
        print("🆕 Starting fresh generation")
    
    print(f"\nReading from: {subset_path}")
    print(f"Saving to: {output_dir}")
    print(f"Rooms per audio: {rooms_per_audio}")
    print("="*60)
    
    start_time = time.time()
    checkpoint_interval = 100  # Save checkpoint every 100 samples
    
    # Iterate through all speakers
    for speaker_dir in sorted(subset_path.iterdir()):
        if not speaker_dir.is_dir():
            continue

        speaker_id = speaker_dir.name

        # Iterate through all chapters for this speaker
        for chapter_dir in sorted(speaker_dir.iterdir()):
            if not chapter_dir.is_dir():
                continue

            chapter_id = chapter_dir.name

            # Get all FLAC files in this chapter
            flac_files = sorted(chapter_dir.glob('*.flac'))

            for flac_file in flac_files:
                # Create unique identifier for this source file
                file_id = f"{speaker_id}_{chapter_id}_{flac_file.stem}"
                
                # Skip if already processed
                if file_id in processed_files:
                    continue
                
                try:
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
                        max_reverb = np.max(np.abs(reverb_audio))
                        max_clean = np.max(np.abs(audio))
                        
                        if max_reverb > 0:
                            reverb_audio = reverb_audio / max_reverb * 0.9
                        else:
                            continue
                        
                        if max_clean > 0:
                            clean_audio_norm = audio / max_clean * 0.9
                        else:
                            continue

                        # Match lengths
                        min_len = min(len(reverb_audio), len(clean_audio_norm))
                        reverb_audio = reverb_audio[:min_len]
                        clean_audio_norm = clean_audio_norm[:min_len]
                        
                        # Check for invalid values
                        if np.isnan(reverb_audio).any() or np.isinf(reverb_audio).any():
                            continue
                        if np.isnan(clean_audio_norm).any() or np.isinf(clean_audio_norm).any():
                            continue
                        
                        # Ensure valid range and type
                        reverb_audio = np.clip(reverb_audio, -1.0, 1.0).astype(np.float32)
                        clean_audio_norm = np.clip(clean_audio_norm, -1.0, 1.0).astype(np.float32)

                        # Save with descriptive names
                        filename = f'{speaker_id}_{chapter_id}_{flac_file.stem}_room{room_idx}.flac'
                        reverb_path = reverb_dir / filename
                        clean_path = clean_dir / filename

                        sf.write(reverb_path, reverb_audio, sr)
                        sf.write(clean_path, clean_audio_norm, sr)

                        sample_idx += 1
                        
                        # Print progress
                        if sample_idx % 100 == 0:
                            elapsed = time.time() - start_time
                            rate = sample_idx / (elapsed / 60)  # samples per minute
                            remaining = (84000 - sample_idx) / rate if sample_idx > 0 else 0
                            
                            print(f"  Generated {sample_idx} samples")
                            print(f"    Rate: {rate:.0f} samples/min")
                            print(f"    Elapsed: {elapsed/60:.1f} min")
                            print(f"    Est. remaining: {remaining:.1f} min")
                        
                        # Save checkpoint periodically
                        if sample_idx % checkpoint_interval == 0:
                            _save_checkpoint(
                                checkpoint_file, 
                                processed_files, 
                                sample_idx
                            )
                    
                    # Mark this source file as processed
                    processed_files.add(file_id)
                    
                except Exception as e:
                    print(f"  Error processing {flac_file.name}: {e}")
                    continue
    
    # Final checkpoint save
    _save_checkpoint(checkpoint_file, processed_files, sample_idx)
    
    # Print final summary
    elapsed_total = time.time() - start_time
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"Total samples generated: {sample_idx}")
    print(f"Total time: {elapsed_total/60:.1f} min ({elapsed_total/3600:.1f} hours)")
    print(f"Average rate: {sample_idx / (elapsed_total/60):.0f} samples/min")
    print(f"Reverb audio: {reverb_dir}")
    print(f"Clean audio: {clean_dir}")
    print(f"{'='*60}")
    
    # Clean up checkpoint file
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print("✓ Removed checkpoint file (generation complete)")


def _save_checkpoint(checkpoint_file, processed_files, sample_count):
    """Save generation progress to checkpoint file"""
    checkpoint_data = {
        'processed_files': list(processed_files),
        'sample_count': sample_count,
        'timestamp': time.time()
    }
    
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f)
    
    print(f"    💾 Checkpoint saved ({sample_count} samples)")

if __name__ == '__main__':
    # Generate large dataset with checkpointing
    create_reverb_from_librispeech(
        librispeech_root='./LibriSpeech',
        output_dir='./dereverb_dataset',
        subset='train-clean-100',  # ~100 hours
        rooms_per_audio=3  # 3 different rooms per utterance
    )
    