from pathlib import Path
import soundfile as sf

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
            file_path.unlink()  # Path's delete method
            deleted_count += 1
    
    if only_clean:
        print(f"Deleting {len(only_clean)} unmatched clean files...")
        for filename in only_clean:
            file_path = clean_dir / filename
            file_path.unlink()  # Path's delete method
            deleted_count += 1
    
    print()
    print("=" * 50)
    print(f"CLEANUP COMPLETE")
    print(f"Final matched pairs: {len(matching)}")
    print(f"Total files deleted: {deleted_count}")
    print("=" * 50)
    
    return len(matching)

def verify_file_integrity(dataset_dir):
    """
    Check if FLAC files can be opened and read
    Delete corrupted files
    """
    reverb_dir = Path(dataset_dir) / 'reverb'
    clean_dir = Path(dataset_dir) / 'clean'
    
    corrupted = []
    
    print("Checking file integrity...")
    
    # Check all files
    all_files = list(reverb_dir.glob('*.flac')) + list(clean_dir.glob('*.flac'))
    
    for i, file_path in enumerate(all_files):
        if i % 100 == 0:
            print(f"  Checked {i}/{len(all_files)} files...")
        
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
    
    print()
    print(f"Found {len(corrupted)} corrupted/empty files")
    
    if corrupted:
        response = input(f"Delete {len(corrupted)} corrupted files? (yes/no): ")
        if response.lower() in ['yes', 'y']:
            for file_path in corrupted:
                file_path.unlink()  # Path's delete method
            print(f"Deleted {len(corrupted)} corrupted files")
    
    return len(corrupted)

# Run both checks
print("STEP 1: Checking for unmatched pairs")
verify_and_cleanup_dataset('./dereverb_dataset', auto_delete=False)

print("\nSTEP 2: Checking file integrity")
verify_file_integrity('./dereverb_dataset')

print("\nSTEP 3: Final verification")
matched_pairs = verify_and_cleanup_dataset('./dereverb_dataset', auto_delete=True)
print(f"\n✓ Dataset ready with {matched_pairs} matched pairs")
