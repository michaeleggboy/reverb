import torch
import torchaudio
from audio_utils import audio_to_spectrogram, spectrogram_to_audio, normalize_db_spectrogram, denormalize_db_spectrogram

TEST_FILE = '/scratch/egbueze.m/reverb_dataset/clean/911_130578_911-130578-0020_room2.flac'
OUTPUT_FILE = 'test_db_roundtrip.wav'

print("Loading audio...")
audio, sr = torchaudio.load(TEST_FILE)
if audio.shape[0] > 1:
    audio = audio[0:1]

print(f"Original audio: shape={audio.shape}, range=[{audio.min():.3f}, {audio.max():.3f}]")

print("\nConverting to dB spectrogram...")
spec_db, phase, ref = audio_to_spectrogram(audio)
print(f"Spectrogram: shape={spec_db.shape}")
print(f"dB range: [{spec_db.min():.1f}, {spec_db.max():.1f}]")
print(f"Reference: {ref:.6f}")

print("\nNormalize -> Denormalize round-trip...")
spec_norm = normalize_db_spectrogram(spec_db)
print(f"Normalized range: [{spec_norm.min():.3f}, {spec_norm.max():.3f}]")

spec_db_back = denormalize_db_spectrogram(spec_norm)
print(f"Denormalized dB range: [{spec_db_back.min():.1f}, {spec_db_back.max():.1f}]")
print(f"dB reconstruction error: {torch.abs(spec_db - spec_db_back).max():.6f}")

print("\nReconstructing audio...")
reconstructed = spectrogram_to_audio(spec_db, phase, ref)
print(f"Reconstructed: shape={reconstructed.shape}, range=[{reconstructed.min():.3f}, {reconstructed.max():.3f}]")

# Save
torchaudio.save(OUTPUT_FILE, reconstructed.unsqueeze(0), sr)
print(f"\nSaved to: {OUTPUT_FILE}")
print("Compare this with the original - they should sound identical!")