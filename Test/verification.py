from pathlib import Path
import soundfile as sf
import json
import time
from tqdm import tqdm


def verify_and_cleanup_dataset(dataset_dir, auto_delete=False):
    """
    Check for matching reverb/clean pairs and optionally delete mismatches
    
    Args:
        dataset_dir: Path to dataset root
        auto_delete: If False, asks for confirmation before deleting
    """
    reverb_dir = Path(dataset_dir) / 'reverb'
    clean_dir = Path(dataset_dir) / 'clean'

    # Check directories exist
    if not reverb_dir.exists():
        print(f"Error: {reverb_dir} doesn't exist")
        return 0
    if not clean_dir.exists():
        print(f"Error: {clean_dir} doesn't exist")
        return 0

    # Get all files with progress bar
    print("Scanning reverb directory...")
    reverb_files = {f.name for f in tqdm(reverb_dir.glob('*.flac'), desc="Reverb files", unit="files")}
    
    print("Scanning clean directory...")
    clean_files = {f.name for f in tqdm(clean_dir.glob('*.flac'), desc="Clean files", unit="files")}

    print(f"\nFound {len(reverb_files)} reverb files")
    print(f"Found {len(clean_files)} clean files")
    print()

    # Find mismatches
    only_reverb = reverb_files - clean_files
    only_clean = clean_files - reverb_files
    matching = reverb_files & clean_files

    print(f"✓ Matching pairs: {len(matching)}")
    print(f"✗ Only in reverb: {len(only_reverb)}")
    print(f"✗ Only in clean: {len(only_clean)}")
    print()

    # If no mismatches, we're done
    if not only_reverb and not only_clean:
        print("✓ All files are properly matched! No cleanup needed.")
        return len(matching)

    # Show examples of mismatches
    if only_reverb:
        print("Examples of unmatched reverb files:")
        for i, filename in enumerate(list(only_reverb)[:5]):
            print(f"  - {filename}")
        if len(only_reverb) > 5:
            print(f"  ... and {len(only_reverb) - 5} more")
        print()

    if only_clean:
        print("Examples of unmatched clean files:")
        for i, filename in enumerate(list(only_clean)[:5]):
            print(f"  - {filename}")
        if len(only_clean) > 5:
            print(f"  ... and {len(only_clean) - 5} more")
        print()

    # Ask for confirmation
    if not auto_delete:
        response = input(f"Delete {len(only_reverb) + len(only_clean)} unmatched files? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Cleanup cancelled.")
            return len(matching)

    # Delete mismatches with progress bar
    deleted_count = 0

    if only_reverb:
        print(f"\nDeleting {len(only_reverb)} unmatched reverb files...")
        for filename in tqdm(only_reverb, desc="Deleting reverb", unit="files"):
            file_path = reverb_dir / filename
            file_path.unlink()
            deleted_count += 1

    if only_clean:
        print(f"\nDeleting {len(only_clean)} unmatched clean files...")
        for filename in tqdm(only_clean, desc="Deleting clean", unit="files"):
            file_path = clean_dir / filename
            file_path.unlink()
            deleted_count += 1

    print()
    print("=" * 50)
    print(f"CLEANUP COMPLETE")
    print(f"Final matched pairs: {len(matching)}")
    print(f"Total files deleted: {deleted_count}")
    print("=" * 50)

    return len(matching)


def verify_file_integrity(dataset_dir, checkpoint_file=None):
    """
    Check if FLAC files can be opened and read
    With checkpointing to resume if interrupted
    
    Args:
        dataset_dir: Path to dataset root
        checkpoint_file: Path to checkpoint file (default: dataset_dir/integrity_checkpoint.json)
    """
    reverb_dir = Path(dataset_dir) / 'reverb'
    clean_dir = Path(dataset_dir) / 'clean'
    
    # Setup checkpoint
    if checkpoint_file is None:
        checkpoint_file = Path(dataset_dir) / 'integrity_checkpoint.json'
    else:
        checkpoint_file = Path(checkpoint_file)
    
    # Get all files to check
    print("Collecting files to verify...")
    reverb_files = sorted(reverb_dir.glob('*.flac'))
    clean_files = sorted(clean_dir.glob('*.flac'))
    all_files = reverb_files + clean_files
    total_files = len(all_files)
    
    print(f"Found {total_files} total files to check")
    print(f"  Reverb: {len(reverb_files)}")
    print(f"  Clean: {len(clean_files)}")
    
    checked_files = set()
    corrupted = []
    
    if checkpoint_file.exists():
        print("\n📂 Found integrity checkpoint, resuming...")
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                checked_files = set(checkpoint_data['checked_files'])
                corrupted = [Path(p) for p in checkpoint_data['corrupted_files']]
            
            print(f"✓ Already checked {len(checked_files)} files")
            print(f"✓ Found {len(corrupted)} corrupted so far")
        except Exception as e:
            print(f"⚠️ Checkpoint error: {e}, starting fresh")
            checked_files = set()
            corrupted = []
    
    print("\nChecking file integrity...")
    
    # Filter out already checked files
    files_to_check = [f for f in all_files if str(f) not in checked_files]
    
    # Progress bar
    pbar = tqdm(
        files_to_check, 
        desc="Verifying files", 
        unit="files",
        initial=len(checked_files),
        total=total_files,
        ncols=100
    )
    
    for file_path in pbar:
        # Update description with current file
        pbar.set_description(f"Checking: {file_path.name[:30]}")
        
        try:
            # Try to read the file
            audio, sr = sf.read(file_path)
            
            # Check if empty or invalid
            if len(audio) == 0:
                corrupted.append(file_path)
                tqdm.write(f"  ✗ Empty file: {file_path.name}")
        
        except Exception as e:
            corrupted.append(file_path)
            tqdm.write(f"  ✗ Corrupted: {file_path.name} - {str(e)[:50]}")
        
        # Mark as checked
        file_str = str(file_path)
        checked_files.add(file_str)
        
        # Update postfix with stats
        pbar.set_postfix({'corrupted': len(corrupted)})
        
        if len(checked_files) % 500 == 0:
            _save_integrity_checkpoint(
                checkpoint_file,
                checked_files,
                [str(p) for p in corrupted]
            )
    
    pbar.close()
    
    print()
    print("=" * 60)
    print("INTEGRITY CHECK COMPLETE")
    print(f"  Total files checked: {len(checked_files)}")
    print(f"  Corrupted files found: {len(corrupted)}")
    print("=" * 60)
    
    if corrupted:
        print("\nCorrupted files:")
        for file_path in corrupted[:10]:
            print(f"  - {file_path.name}")
        if len(corrupted) > 10:
            print(f"  ... and {len(corrupted) - 10} more")
        
        response = input(f"\nDelete {len(corrupted)} corrupted files? (yes/no): ")
        if response.lower() in ['yes', 'y']:
            print("\nDeleting corrupted files...")
            for file_path in tqdm(corrupted, desc="Deleting", unit="files"):
                file_path.unlink()
            print(f"✓ Deleted {len(corrupted)} corrupted files")
            
            # Also delete their pairs
            print("\nCleaning up orphaned pairs...")
            verify_and_cleanup_dataset(dataset_dir, auto_delete=True)
    else:
        print("✓ No corrupted files found!")
    
    # Clean up checkpoint on completion
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print("✓ Removed integrity checkpoint (check complete)")
    
    return len(corrupted)


def _save_integrity_checkpoint(checkpoint_file, checked_files, corrupted_files):
    """
    Save integrity check progress
    
    Args:
        checkpoint_file: Path to checkpoint
        checked_files: Set of file paths already checked
        corrupted_files: List of corrupted file paths
    """
    checkpoint_data = {
        'checked_files': sorted(list(checked_files)),
        'corrupted_files': corrupted_files,
        'timestamp': time.time()
    }
    
    # Atomic write
    temp_file = checkpoint_file.with_suffix('.tmp')
    
    try:
        with open(temp_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        temp_file.replace(checkpoint_file)
        
    except Exception as e:
        tqdm.write(f"    ⚠️ Checkpoint save failed: {e}")
        if temp_file.exists():
            temp_file.unlink()


def quick_verify(dataset_dir):
    """
    Ultra-fast check - only verifies matching pairs
    Recommended for regular use (< 1 second)
    
    Args:
        dataset_dir: Path to dataset root
    
    Returns:
        Number of matched pairs
    """
    reverb_dir = Path(dataset_dir) / 'reverb'
    clean_dir = Path(dataset_dir) / 'clean'
    
    if not reverb_dir.exists() or not clean_dir.exists():
        print("Error: Dataset directories not found")
        return 0
    
    print("Scanning directories...")
    reverb = set(f.name for f in tqdm(
        reverb_dir.glob('*.flac'), 
        desc="Reverb", 
        unit="files",
        leave=False
    ))
    clean = set(f.name for f in tqdm(
        clean_dir.glob('*.flac'), 
        desc="Clean", 
        unit="files",
        leave=False
    ))
    
    matching = reverb & clean
    mismatched = reverb ^ clean  # Symmetric difference
    
    print(f"✓ Matched pairs: {len(matching)}")
    
    if mismatched:
        print(f"⚠️ Mismatched files: {len(mismatched)}")
        print("  Run verify_and_cleanup_dataset() for details")
    else:
        print("✓ All files properly paired!")
    
    return len(matching)


def count_samples_by_rooms(dataset_dir):
    """
    Count how many samples per room configuration exist
    
    Args:
        dataset_dir: Path to dataset root
    
    Returns:
        Dict of room counts
    """
    reverb_dir = Path(dataset_dir) / 'reverb'
    
    if not reverb_dir.exists():
        print("Error: Reverb directory not found")
        return {}
    
    print("Analyzing room distribution...")
    
    room_counts = {}
    files = list(reverb_dir.glob('*.flac'))
    
    for file_path in tqdm(files, desc="Counting rooms", unit="files"):
        # Extract room number from filename
        # Format: speaker_chapter_file_roomX.flac
        filename = file_path.stem
        if '_room' in filename:
            room_num = filename.split('_room')[-1]
            room_counts[f'room{room_num}'] = room_counts.get(f'room{room_num}', 0) + 1
    
    print("\nRoom distribution:")
    for room, count in sorted(room_counts.items()):
        print(f"  {room}: {count} samples")
    
    return room_counts


if __name__ == '__main__':
    dataset_path = '/scratch/egbueze.m/reverb_dataset'
    
    print("="*60)
    print("DATASET VERIFICATION WITH TQDM")
    print("="*60)
    
    # STEP 1: Quick pair matching check
    print("\nSTEP 1: Quick verification (checking pairs)")
    print("-"*60)
    matched = quick_verify(dataset_path)
    
    # STEP 2: Cleanup mismatches if any found
    if matched == 0:
        print("\nSTEP 2: Detailed pair check and cleanup")
        print("-"*60)
        matched = verify_and_cleanup_dataset(dataset_path, auto_delete=False)
    
    # STEP 3: Optional deep integrity check
    print("\nSTEP 3: Deep integrity check (optional, slower)")
    print("-"*60)
    response = input("Run deep integrity check? This reads all files. (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        print("\nRunning deep integrity check...")
        print("(This can be interrupted and resumed)")
        corrupted_count = verify_file_integrity(dataset_path)
        
        if corrupted_count > 0:
            print("\n⚠️ Found corruption, running final pair check...")
            matched = verify_and_cleanup_dataset(dataset_path, auto_delete=True)
    else:
        print("Skipped deep integrity check")
    
    # STEP 4: Room distribution analysis
    print("\nSTEP 4: Room distribution analysis")
    print("-"*60)
    response = input("Analyze room distribution? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        room_counts = count_samples_by_rooms(dataset_path)
    
    # FINAL SUMMARY
    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)
    print(f"✓ Dataset ready with {matched} matched pairs")
    print("="*60)
