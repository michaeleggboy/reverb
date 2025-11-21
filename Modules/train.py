from pathlib import Path
import time
from fast_dataset import PrecomputedDataset
import unet
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np


def train_model(
    train_dataset,
    val_dataset,
    num_epochs=50,
    batch_size=32,
    learning_rate=1e-3,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    checkpoint_dir='checkpoints',
    save_every=5,
    resume_from=None,
    use_scheduler=True,
    accumulation_steps=1,
    use_amp=True,
):
    """Train the U-Net model with pre-computed spectrograms"""
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    # Print GPU info
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("\nCreating DataLoaders...")
    
    # Optimized for pre-computed data
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True if device == 'cuda' else False,
        persistent_workers=True if device == 'cuda' else False,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True if device == 'cuda' else False,
        persistent_workers=True if device == 'cuda' else False,
        prefetch_factor=2
    )

    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")

    # Initialize model
    print("\nInitializing model...")
    model = unet.UNet(in_channels=1, out_channels=1).to(device)
    
    # if hasattr(torch, 'compile'):
    #     print("Compiling model (takes 30-60 seconds)...")
    #     model = torch.compile(model, mode="reduce-overhead")
    
    #     # Trigger compilation with dummy forward pass
    #     dummy = torch.randn(1, 1, 256, 256).to(device)
    #     with torch.no_grad():
    #         _ = model(dummy)
    #     del dummy
    #     torch.cuda.empty_cache()
    #     print("✓ Model compiled")
    # else:
    #     print("torch.compile not available (need PyTorch 2.0+)")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )

    scaler = GradScaler("cuda") if use_amp and device == 'cuda' else None
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    ) if use_scheduler else None

    best_val_loss = float('inf')
    start_epoch = 0

    # Check for existing checkpoints
    latest_checkpoint = checkpoint_dir / 'latest_checkpoint.pth'
    if latest_checkpoint.exists():
        print(f"\n📂 Loading checkpoint from: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"✓ Resumed from epoch {checkpoint['epoch'] + 1}")

    print("\n" + "="*60)
    print(f"Training from epoch {start_epoch + 1} to {num_epochs}")
    print(f"Batch size: {batch_size}, LR: {learning_rate}")
    print("="*60 + "\n")
    
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✓ cuDNN benchmark enabled\n")

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
    
        # Training
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train")
        
        for batch_idx, (reverb_spec, clean_spec) in enumerate(train_pbar):
            reverb_spec = reverb_spec.to(device, non_blocking=True)
            clean_spec = clean_spec.to(device, non_blocking=True)
            
            # Ensure correct shape [B, 1, H, W]
            if reverb_spec.dim() == 3:
                reverb_spec = reverb_spec.unsqueeze(1)
            if clean_spec.dim() == 3:
                clean_spec = clean_spec.unsqueeze(1)
            
            # Don't zero grad every batch when accumulating
            if batch_idx % accumulation_steps == 0:
                optimizer.zero_grad()
            
            if scaler:
                with autocast("cuda"):
                    pred_spec = model(reverb_spec)
                    loss = criterion(pred_spec, clean_spec)
                    loss = loss / accumulation_steps
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                pred_spec = model(reverb_spec)
                loss = criterion(pred_spec, clean_spec)
                loss = loss / accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            
            train_loss += loss.item() * accumulation_steps
            train_pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Handle remaining gradients
        if (len(train_loader) % accumulation_steps) != 0:
            if scaler and scaler.get_scale() != 0:
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()   

        # Validation
        model.eval()
        val_loss = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Val")
            
        with torch.no_grad():
            for reverb_spec, clean_spec in val_pbar:
                reverb_spec = reverb_spec.to(device, non_blocking=True)
                clean_spec = clean_spec.to(device, non_blocking=True)
                    
                if reverb_spec.dim() == 3:
                    reverb_spec = reverb_spec.unsqueeze(1)
                if clean_spec.dim() == 3:
                    clean_spec = clean_spec.unsqueeze(1)
                    
                pred_spec = model(reverb_spec)
                loss = criterion(pred_spec, clean_spec)
                val_loss += loss.item()
                val_pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
            
        avg_val_loss = val_loss / len(val_loader)
        
        # Update scheduler
        if scheduler:
            scheduler.step(avg_val_loss)
            print(f"  ✓ LR adjusted to: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(f"[Epoch {epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Time: {epoch_time:.1f}s | "
              f"Speed: {len(train_loader)/epoch_time:.1f} batch/s")
        
        # Save checkpoints
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }
        
        torch.save(checkpoint_data, checkpoint_dir / 'latest_checkpoint.pth')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint_data, checkpoint_dir / 'best_model.pth')
            print(f"  ✓ New best model saved (val_loss: {avg_val_loss:.4f})")
        
        if (epoch + 1) % save_every == 0:
            torch.save(checkpoint_data, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth')
            print(f"  ✓ Checkpoint saved for epoch {epoch+1}")

    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print("="*60)
    
    return model


if __name__ == '__main__':
    # Check if pre-computed specs exist
    spec_dir = Path('/scratch/egbueze.m/precomputed_specs')
    
    if not spec_dir.exists() or len(list(spec_dir.glob('*.pt'))) == 0:
        print("❌ No pre-computed spectrograms found!")
        print("Run: python precompute_specs.py first!")
        exit(1)
    
    # Load pre-computed dataset
    print("Loading pre-computed spectrograms...")
    full_dataset = PrecomputedDataset(spec_dir)
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"✓ Train: {train_size} samples")
    print(f"✓ Val: {val_size} samples")
    
    model = train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=50,
        batch_size=64,
        learning_rate=2e-4,
        device='cuda',
        checkpoint_dir='/scratch/egbueze.m/checkpoints',
        save_every=10,
        accumulation_steps=2,
        use_amp=False,
    )
