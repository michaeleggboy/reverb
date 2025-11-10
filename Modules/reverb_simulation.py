from pathlib import Path
import json
import time
import pyroomacoustics as pra
import numpy as np
import soundfile as sf


def create_reverb_from_librispeech(
    librispeech_root,
    output_dir,
    subset='train-clean-100',
    rooms_per_audio=3,
    checkpoint_file=None
):
    """
    Generate reverberant versions from LibriSpeech clean data
    
    Features:
    - Automatic checkpointing and resume capability
    - Progress tracking with time estimates
    - Error handling and validation
    - Atomic checkpoint saves to prevent corruption
    
    Args:
        librispeech_root: Path to LibriSpeech root (contains train-clean-100/, etc.)
        output_dir: Where to save reverb/clean pairs
        subset: Which LibriSpeech subset to use ('train-clean-100', 'dev-clean', etc.)
        rooms_per_audio: Number of different room configs per audio file
        checkpoint_file: Path to save/load progress (default: output_dir/generation_checkpoint.json)
    """

    # ===== SETUP PATHS =====
    subset_path = Path(librispeech_root) / subset
    reverb_dir = Path(output_dir) / 'reverb'
    clean_dir = Path(output_dir) / 'clean'
    reverb_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate LibriSpeech path exists
    if not subset_path.exists():
        raise FileNotFoundError(f"LibriSpeech subset not found: {subset_path}")
    
    # ===== CHECKPOINT SETUP =====
    if checkpoint_file is None:
        checkpoint_file = Path(output_dir) / 'generation_checkpoint.json'
    else:
        checkpoint_file = Path(checkpoint_file)
    
    # ===== COUNT TOTAL SOURCE FILES =====
    print("Counting source files...")
    total_source_files = 0
    for speaker_dir in subset_path.iterdir():
        if not speaker_dir.is_dir():
            continue
        for chapter_dir in speaker_dir.iterdir():
            if not chapter_dir.is_dir():
                continue
            total_source_files += len(list(chapter_dir.glob('*.flac')))
    
    expected_total_samples = total_source_files * rooms_per_audio
    
    print(f"Found {total_source_files} source audio files")
    print(f"Expected total samples: {expected_total_samples} ({rooms_per_audio} rooms per file)")
    
    # ===== LOAD CHECKPOINT IF EXISTS =====
    processed_files = set()
    sample_idx = 0
    
    if checkpoint_file.exists():
        print(f"\n📂 Found checkpoint: {checkpoint_file}")
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                processed_files = set(checkpoint_data['processed_files'])
                sample_idx = len(processed_files) * rooms_per_audio
            print(f"✓ Resuming from {sample_idx} samples")
            print(f"✓ Already processed {len(processed_files)} source files")
        except Exception as e:
            print(f"⚠️ Checkpoint file corrupted: {e}")
            print("Starting fresh...")
            processed_files = set()
            sample_idx = 0
    else:
        print("\n🆕 Starting fresh generation")
    
    print(f"\nReading from: {subset_path}")
    print(f"Saving to: {output_dir}")
    print(f"Rooms per audio: {rooms_per_audio}")
    print("="*60)
    
    start_time = time.time()
    
    # ===== MAIN GENERATION LOOP =====
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
                
                # ===== SKIP IF ALREADY PROCESSED =====
                if file_id in processed_files:
                    continue
                
                try:
                    # Load clean audio
                    audio, sr = sf.read(flac_file)

                    # Convert to mono if stereo
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=1)
                    
                    samples_this_file = 0

                    # Generate multiple reverberant versions
                    for room_idx in range(rooms_per_audio):
                        # Random room parameters
                        room_dim = np.random.uniform([3, 3, 2.5], [10, 10, 4])
                        rt60 = np.random.uniform(0.2, 1.0)

                        # Create room with image source method
                        room = pra.ShoeBox(
                            room_dim,
                            fs=sr,
                            materials=pra.Material(rt60),
                            max_order=15  # Number of wall reflections
                        )

                        # Random source and microphone positions
                        # Ensure positions are within room bounds with 0.5m margin
                        source_pos = np.random.uniform(
                            [0.5, 0.5, 0.5],
                            room_dim - [0.5, 0.5, 0.5]
                        )
                        mic_pos = np.random.uniform(
                            [0.5, 0.5, 0.5],
                            room_dim - [0.5, 0.5, 0.5]
                        )

                        # Add source and microphone to room
                        room.add_source(source_pos, signal=audio)
                        room.add_microphone(mic_pos)
                        
                        # Simulate room acoustics
                        room.simulate()
                        reverb_audio = room.mic_array.signals[0, :]

                        # ===== NORMALIZATION =====
                        # Normalize both reverb and clean to prevent clipping
                        max_reverb = np.max(np.abs(reverb_audio))
                        max_clean = np.max(np.abs(audio))
                        
                        if max_reverb > 0:
                            reverb_audio = reverb_audio / max_reverb * 0.9
                        else:
                            print("  ⚠️ Silent reverb audio, skipping")
                            continue
                        
                        if max_clean > 0:
                            clean_audio_norm = audio / max_clean * 0.9
                        else:
                            print("  ⚠️ Silent clean audio, skipping")
                            continue

                        # ===== MATCH LENGTHS =====
                        # Reverb tail might be longer than original
                        min_len = min(len(reverb_audio), len(clean_audio_norm))
                        reverb_audio = reverb_audio[:min_len]
                        clean_audio_norm = clean_audio_norm[:min_len]
                        
                        # ===== VALIDATION =====
                        if np.isnan(reverb_audio).any() or np.isinf(reverb_audio).any():
                            print("  ⚠️ NaN/Inf in reverb, skipping")
                            continue
                        if np.isnan(clean_audio_norm).any() or np.isinf(clean_audio_norm).any():
                            print("  ⚠️ NaN/Inf in clean, skipping")
                            continue
                        
                        reverb_audio = np.clip(reverb_audio, -1.0, 1.0).astype(np.float32)
                        clean_audio_norm = np.clip(clean_audio_norm, -1.0, 1.0).astype(np.float32)

                        # ===== SAVE FILES =====
                        filename = f'{speaker_id}_{chapter_id}_{flac_file.stem}_room{room_idx}.flac'
                        reverb_path = reverb_dir / filename
                        clean_path = clean_dir / filename

                        sf.write(reverb_path, reverb_audio, sr)
                        sf.write(clean_path, clean_audio_norm, sr)
                        
                        samples_this_file += 1
                    
                    # ===== UPDATE PROGRESS AFTER FILE COMPLETE =====
                    # This ensures checkpoint consistency
                    sample_idx += samples_this_file
                    processed_files.add(file_id)
                    
                    # ===== SAVE CHECKPOINT PERIODICALLY =====
                    # This balances checkpoint frequency vs I/O overhead
                    if len(processed_files) % 10 == 0:
                        _save_checkpoint(
                            checkpoint_file,
                            processed_files,
                            sample_idx
                        )
                    
                    # ===== PROGRESS REPORTING =====
                    if sample_idx % 100 == 0 or sample_idx % 300 == 0:
                        elapsed = time.time() - start_time
                        rate = sample_idx / (elapsed / 60) if elapsed > 0 else 0
                        
                        if rate > 0 and expected_total_samples > 0:
                            remaining = (expected_total_samples - sample_idx) / rate
                            progress_pct = 100 * sample_idx / expected_total_samples
                        else:
                            remaining = 0
                            progress_pct = 0
                        
                        print(f"\n📊 Progress: {sample_idx}/{expected_total_samples} samples ({progress_pct:.1f}%)")
                        print(f"   Rate: {rate:.0f} samples/min")
                        print(f"   Elapsed: {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")
                        if remaining > 0:
                            print(f"   Est. remaining: {remaining:.1f} min ({remaining/60:.1f} hours)")
                
                except Exception as e:
                    print(f"  ✗ Error processing {flac_file.name}: {e}")
                    continue
    
    # ===== FINAL CHECKPOINT SAVE =====
    _save_checkpoint(checkpoint_file, processed_files, sample_idx)
    
    # ===== FINAL SUMMARY =====
    elapsed_total = time.time() - start_time
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total samples generated: {sample_idx}")
    print(f"Source files processed: {len(processed_files)}")
    print(f"Total time: {elapsed_total/60:.1f} min ({elapsed_total/3600:.2f} hours)")
    print(f"Average rate: {sample_idx / (elapsed_total/60):.0f} samples/min")
    print(f"Reverb audio: {reverb_dir}")
    print(f"Clean audio: {clean_dir}")
    print(f"{'='*60}")
    
    # Clean up checkpoint file on successful completion
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print("✓ Removed checkpoint file (generation complete)")


def _save_checkpoint(checkpoint_file, processed_files, sample_count):
    """
    Save generation progress to checkpoint file
    
    FIX: Uses atomic write to prevent corruption
    Writes to temp file first, then renames (atomic operation)
    """
    checkpoint_data = {
        'processed_files': sorted(list(processed_files)),  # Sorted for readability
        'sample_count': sample_count,
        'timestamp': time.time()
    }
    
    # This prevents corruption if crash happens during write
    temp_file = checkpoint_file.with_suffix('.tmp')
    
    try:
        with open(temp_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        # Atomic rename (won't corrupt existing checkpoint)
        temp_file.replace(checkpoint_file)
        
    except Exception as e:
        print(f"    ⚠️ Failed to save checkpoint: {e}")
        if temp_file.exists():
            temp_file.unlink()


if __name__ == '__main__':
    # Generate large dataset with checkpointing
    create_reverb_from_librispeech(
        librispeech_root='./LibriSpeech',
        output_dir='./dereverb_dataset',
        subset='train-clean-100',  # ~100 hours
        rooms_per_audio=3  # 3 different rooms per utterance
    )
    