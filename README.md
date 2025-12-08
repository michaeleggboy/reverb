# Audio Dereverberation using U-Net

Deep learning system for removing reverberation from speech audio using U-Net architecture and synthetic training data.

## Overview

This project removes echo and reverberation from speech recordings using a convolutional neural network trained on spectrograms. The model learns to predict clean speech from reverberant input by processing magnitude spectrograms while preserving phase information.

## Features

- Synthetic reverberant data generation using room acoustics simulation
- U-Net model with skip connections for spectrogram processing
- Automatic checkpointing for long-running tasks
- Batch inference with format control
- Comprehensive testing suite

## Quick Start

### Installation

```bash
pip install torch torchaudio pyroomacoustics soundfile numpy
```

### Generate Training Data

```python
from data_generation import create_reverb_from_librispeech

create_reverb_from_librispeech(
    librispeech_root='./LibriSpeech',
    output_dir='./dataset',
    subset='train-clean-100',
    rooms_per_audio=3
)
```

### Train Model

```python
from train import train_model
from dataset import DereverbDataset
from torch.utils.data import random_split

# Load dataset
dataset = DereverbDataset('dataset/reverb', 'dataset/clean')
train_set, val_set = random_split(dataset, [0.8, 0.2])

# Train
model = train_model(
    train_dataset=train_set,
    val_dataset=val_set,
    num_epochs=40,
    batch_size=32,
    device='cuda'
)
```

### Remove Reverb

```python
from inference import dereverb_audio

dereverb_audio(
    input_audio_path='reverberant.flac',
    output_audio_path='clean.flac',
    model_path='checkpoints/best_model.pth'
)
```

## Architecture

**U-Net Model:**
- Input: 1040x512 magnitude spectrogram
- Encoder: 4 downsampling blocks (64→128→256→512 features)
- Bottleneck: 1024 features
- Decoder: 4 upsampling blocks with skip connections
- Output: Clean magnitude with Sigmoid activation
    + Mask
    + Residual + Attention

**Audio Processing:**
- STFT: n_fft=2048, hop_length=512, Hann window
- Phase preservation from original audio
- db max reference magnitude rescaling

## Model

best_model_80db.pth:
 - [-80, 0] dB scaling model
 - direct output
 - val loss: 0.0912 (epoch 30)

best_model_120db.pth:
 - [-120, 0] dB scaling model
 - direct output
 - val loss: 0.0604 (epoch 17)

best_model_120db_residual.pth:
 - [-120, 0] dB scaling model
 - residual output + attention
 - val loss: 0.0836 (epoch 26)
