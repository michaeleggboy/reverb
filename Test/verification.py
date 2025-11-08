# ==========================================
# VERIFY_DATASET.PY - With Checkpointing for Integrity Check
# ==========================================

from pathlib import Path
import soundfile as sf
import json


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

    # Get all files
    reverb_files = {f.name for f in reverb_dir.glob('*.flac')}
    clean_files = {f.name for f in clean_dir.glob('*.flac')}

    print(f"Found {len(reverb_files)} reverb files")
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

    # Delete mismatches
    deleted_count = 0

    if only_reverb:
        print(f"Deleting {len(only_reverb)} unmatched reverb files...")
        for filename in only_reverb:
            file_path = reverb_dir / filename
            file_path.unlink()
            deleted_count += 1

    if only_clean:
        print(f"Deleting {len(only_clean)} unmatched clean files...")
        for filename in only_clean:
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
    all_files = list(reverb_dir.glob('*.flac')) + list(clean_dir.glob('*.flac'))
    total_files = len(all_files)
    
    print(f"Checking integrity of {total_files} files...")
    
    # ===== LOAD CHECKPOINT IF EXISTS =====
    checked_files = set()
    corrupted = []
    
    if checkpoint_file.exists():
        print(f"📂 Found integrity checkpoint, resuming...")
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
    
    # ===== CHECK FILES =====
    files_checked_this_session = 0
    
    for i, file_path in enumerate(all_files):
        # Convert to string for set comparison
        file_str = str(file_path)
        
        # Skip if already checked
        if file_str in checked_files:
            continue
        
        # Progress update
        total_checked = len(checked_files) + files_checked_this_session
        if total_checked % 100 == 0 and total_checked > 0:
            progress_pct = 100 * total_checked / total_files
            print(f"  Checked {total_checked}/{total_files} files ({progress_pct:.1f}%)...")
        
        try:
            # Try to read the file
            audio, sr = sf.read(file_path)
            
            # Check if empty or invalid
            if len(audio) == 0:
                print(f"  Empty file: {file_path.name}")
                corrupted.append(file_path)
        
        except Exception as e:
            print(f"  Corrupted: {file_path.name} - {e}")
            corrupted.append(file_path)
        
        # Mark as checked
        checked_files.add(file_str)
        files_checked_this_session += 1
        
        # ===== SAVE CHECKPOINT EVERY 500 FILES =====
        if len(checked_files) % 500 == 0:
            _save_integrity_checkpoint(
                checkpoint_file,
                checked_files,
                [str(p) for p in corrupted]
            )
    
    # ===== FINAL RESULTS =====
    print()
    print(f"Checked {len(checked_files)} files total")
    print(f"Found {len(corrupted)} corrupted/empty files")
    
    if corrupted:
        print("\nCorrupted files:")
        for file_path in corrupted[:10]:
            print(f"  - {file_path.name}")
        if len(corrupted) > 10:
            print(f"  ... and {len(corrupted) - 10} more")
        
        response = input(f"\nDelete {len(corrupted)} corrupted files? (yes/no): ")
        if response.lower() in ['yes', 'y']:
            for file_path in corrupted:
                file_path.unlink()
            print(f"✓ Deleted {len(corrupted)} corrupted files")
            
            # Also delete their pairs
            print("\nCleaning up orphaned pairs...")
            verify_and_cleanup_dataset(dataset_dir, auto_delete=True)
    
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
        'timestamp': time.time() if 'time' in dir() else 0
    }
    
    # Atomic write
    temp_file = checkpoint_file.with_suffix('.tmp')
    
    try:
        with open(temp_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        temp_file.replace(checkpoint_file)
        
    except Exception as e:
        print(f"    ⚠️ Checkpoint save failed: {e}")
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
    
    reverb = set(f.name for f in reverb_dir.glob('*.flac'))
    clean = set(f.name for f in clean_dir.glob('*.flac'))
    
    matching = reverb & clean
    mismatched = reverb ^ clean  # Symmetric difference
    
    print(f"✓ Matched pairs: {len(matching)}")
    
    if mismatched:
        print(f"⚠️ Mismatched files: {len(mismatched)}")
        print("  Run verify_and_cleanup_dataset() for details")
    else:
        print("✓ All files properly paired!")
    
    return len(matching)


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == '__main__':
    import time  # For checkpoint timestamps
    
    dataset_path = './drive/MyDrive/dereverb_dataset'
    
    print("="*60)
    print("DATASET VERIFICATION")
    print("="*60)
    
    # STEP 1: Quick pair matching check (1-2 seconds)
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
    response = input("Run deep integrity check? This reads all files (~30-60 sec). (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        print("\nRunning deep integrity check...")
        print("(This can be interrupted and resumed)")
        corrupted_count = verify_file_integrity(dataset_path)
        
        if corrupted_count > 0:
            print("\n⚠️ Found corruption, running final pair check...")
            matched = verify_and_cleanup_dataset(dataset_path, auto_delete=True)
    else:
        print("Skipped deep integrity check")
    
    # FINAL SUMMARY
    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)
    print(f"✓ Dataset ready with {matched} matched pairs")
    print("="*60)