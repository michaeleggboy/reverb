from pathlib import Path
import json
import time
import pyroomacoustics as pra
import numpy as np
import soundfile as sf
from tqdm import tqdm


def create_reverb_from_librispeech(
    librispeech_root,
    output_dir,
    subset='train-clean-100',
    rooms_per_audio=3,
    checkpoint_file=None,
    frequency_dependent_rt60=True
):
    """
    Generate reverberant versions from LibriSpeech clean data
    """

    subset_path = Path(librispeech_root) / subset
    reverb_dir = Path(output_dir) / 'reverb'
    clean_dir = Path(output_dir) / 'clean'
    reverb_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    
    if not subset_path.exists():
        raise FileNotFoundError(f"LibriSpeech subset not found: {subset_path}")
    
    if checkpoint_file is None:
        checkpoint_file = Path(output_dir) / 'generation_checkpoint.json'
    else:
        checkpoint_file = Path(checkpoint_file)
    
    print("Counting source files...")
    total_source_files = 0
    all_files = []
    for speaker_dir in subset_path.iterdir():
        if not speaker_dir.is_dir():
            continue
        for chapter_dir in speaker_dir.iterdir():
            if not chapter_dir.is_dir():
                continue
            for flac_file in chapter_dir.glob('*.flac'):
                all_files.append((speaker_dir.name, chapter_dir.name, flac_file))
                total_source_files += 1
    
    expected_total_samples = total_source_files * rooms_per_audio
    
    print(f"Found {total_source_files} source audio files")
    print(f"Expected total samples: {expected_total_samples}")
    print(f"Settings: Frequency Dependent rt60 | Adaptive rt60 | rooms_per_audio={rooms_per_audio}")
    
    processed_files = set()
    sample_idx = 0
    
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                processed_files = set(checkpoint_data['processed_files'])
                sample_idx = len(processed_files) * rooms_per_audio
            print(f"✓ Resuming from {sample_idx} samples ({len(processed_files)} files)")
        except Exception:
            processed_files = set()
            sample_idx = 0
    
    print("="*60)
    
    # Main progress bar
    pbar = tqdm(total=expected_total_samples, initial=sample_idx, 
                desc="Generating reverb", unit="samples", ncols=100)
    
    # Stats tracking
    skipped_count = 0
    
    for speaker_id, chapter_id, flac_file in all_files:
        file_id = f"{speaker_id}_{chapter_id}_{flac_file.stem}"
        
        if file_id in processed_files:
            continue
        
        try:
            audio, sr = sf.read(flac_file)
            
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            audio = audio.astype(np.float32)
            
            # Check audio quality
            if np.max(np.abs(audio)) < 0.001:
                skipped_count += 1
                continue
            
            samples_this_file = 0

            for room_idx in range(rooms_per_audio):
                room_dim = np.random.uniform([3, 3, 2.5], [10, 10, 4])
                room_volume = np.prod(room_dim)
                volume_factor = np.clip(room_volume / 400, 0.0, 1.0)

                rt60_min = 0.2 + 0.2 * volume_factor
                rt60_max = 0.5 + 0.5 * volume_factor
                rt60 = np.random.uniform(rt60_min, rt60_max)
                rt60_base = np.clip(rt60, 0.15, 1.2)       

                if frequency_dependent_rt60:
                    try:
                        tilt = np.random.uniform(-0.08, 0.08)
                        
                        rt60_bands = np.array([
                            rt60_base * (1.0 + tilt),        # 125 Hz
                            rt60_base * (1.0 + tilt*0.6),    # 250 Hz
                            rt60_base,                       # 500 Hz (reference)
                            rt60_base * (1.0 - tilt*0.3),    # 1000 Hz
                            rt60_base * (1.0 - tilt*0.6),    # 2000 Hz
                            rt60_base * (1.0 - tilt),        # 4000 Hz
                        ])
                        rt60_bands = np.clip(rt60_bands, 0.1, 1.2)
                        
                        absorption_coeffs = []
                        for rt60 in rt60_bands:
                            try:
                                e_abs, _ = pra.inverse_sabine(rt60, room_dim)
                                e_abs = np.clip(e_abs, 0.01, 0.98)
                                absorption_coeffs.append(e_abs)
                            except:
                                materials = pra.Material(rt60_base)
                                break
                        else:
                            material_dict = {
                                'description': 'Custom frequency-dependent material',
                                'coeffs': absorption_coeffs,
                                'center_freqs': [125, 250, 500, 1000, 2000, 4000]
                            }
                            materials = pra.Material(energy_absorption=material_dict)
                            
                    except Exception as e:
                        materials = pra.Material(rt60_base)
                else:
                    materials = pra.Material(rt60_base)
                
                # Create room with optimizations
                room = pra.ShoeBox(
                    room_dim,
                    fs=sr,
                    materials=materials,
                    max_order=15
                )

                # Generate positions
                source_pos = np.random.uniform([0.5, 0.5, 0.5], room_dim - [0.5, 0.5, 0.5])
                mic_pos = np.random.uniform([0.5, 0.5, 0.5], room_dim - [0.5, 0.5, 0.5])
                
                # Ensure minimum distance
                min_distance = 0.5
                attempts = 0
                while np.linalg.norm(mic_pos - source_pos) < min_distance and attempts < 50:
                    mic_pos = np.random.uniform([0.5, 0.5, 0.5], room_dim - [0.5, 0.5, 0.5])
                    attempts += 1

                room.add_source(source_pos, signal=audio)
                room.add_microphone(mic_pos)
                
                room.simulate()
                reverb_audio = room.mic_array.signals[0, :]

                # Skip if NaN/Inf
                if np.isnan(reverb_audio).any() or np.isinf(reverb_audio).any():
                    skipped_count += 1
                    continue

                # DRR-preserving normalization
                max_clean = np.max(np.abs(audio))
                if max_clean > 0:
                    target_peak = 0.9
                    scale_multiplier = target_peak / max_clean
                    
                    clean_audio_norm = audio * scale_multiplier
                    reverb_audio_scaled = reverb_audio * scale_multiplier
                    reverb_audio = np.clip(reverb_audio_scaled, -1.0, 1.0)
                else:
                    skipped_count += 1
                    continue

                # Ensure same length
                min_len = min(len(reverb_audio), len(clean_audio_norm))
                reverb_audio = reverb_audio[:min_len]
                clean_audio_norm = clean_audio_norm[:min_len]
                
                # Final type conversion
                reverb_audio = reverb_audio.astype(np.float32)
                clean_audio_norm = clean_audio_norm.astype(np.float32)

                # Save files
                filename = f'{speaker_id}_{chapter_id}_{flac_file.stem}_room{room_idx}.flac'
                reverb_path = reverb_dir / filename
                clean_path = clean_dir / filename

                sf.write(reverb_path, reverb_audio, sr)
                sf.write(clean_path, clean_audio_norm, sr)
                
                samples_this_file += 1
            
            sample_idx += samples_this_file
            processed_files.add(file_id)
            
            # Update progress bar
            pbar.update(samples_this_file)
            pbar.set_postfix({'skipped': skipped_count, 'files': len(processed_files)})
            
            # Checkpoint saving
            if len(processed_files) % 10 == 0:
                _save_checkpoint(checkpoint_file, processed_files, sample_idx)
        
        except Exception as e:
            pbar.set_postfix_str(f"Error: {flac_file.name[:20]}")
            continue
    
    pbar.close()
    
    _save_checkpoint(checkpoint_file, processed_files, sample_idx)
    
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"Total samples generated: {sample_idx}")
    print(f"Source files processed: {len(processed_files)}")
    print(f"Problematic configs skipped: {skipped_count}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")
    
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print("✓ Removed checkpoint file")


def _save_checkpoint(checkpoint_file, processed_files, sample_count):
    """Save generation progress to checkpoint file"""
    checkpoint_data = {
        'processed_files': sorted(list(processed_files)),
        'sample_count': sample_count,
        'timestamp': time.time()
    }
    
    temp_file = checkpoint_file.with_suffix('.tmp')
    
    try:
        with open(temp_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        temp_file.replace(checkpoint_file)
    except Exception:
        pass


if __name__ == '__main__':
    create_reverb_from_librispeech(
        librispeech_root='/scratch/egbueze.m/librispeech/LibriSpeech',
        output_dir='/scratch/egbueze.m/reverb_dataset',
        subset='train-clean-100',
        rooms_per_audio=3,
        frequency_dependent_rt60=True
    )
