from precomputed_dataset import PrecomputedDataset
from spectral_loss import SpectralLoss
from unet import UNet
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast
from tqdm import tqdm


def train_model(
    train_dataset,
    val_dataset,
    num_epochs=50,
    batch_size=32,
    learning_rate=1e-3,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    checkpoint_dir='checkpoints',
    save_every=5,
    use_scheduler=True,
    accumulation_steps=1,
    use_amp=True,
    resume_epoch=None,
    force_resume=False
):
    """Train the U-Net model with pre-computed spectrograms"""
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    # Print GPU info
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("\nCreating DataLoaders...")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=device == 'cuda',
        persistent_workers=device == 'cuda',
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device == 'cuda',
        persistent_workers=device == 'cuda',
        prefetch_factor=2
    )

    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")

    # Initialize model
    print("\nInitializing model...")
    model = UNet(in_channels=1, out_channels=1).to(device)

    criterion = SpectralLoss(adaptive_weights=False, use_perceptual=True).to(device)
    optimizer = torch.optim.AdamW([
        {
            'params': model.parameters(), 
            'lr': learning_rate, 
            'weight_decay': 1e-4, 
            'betas': (0.9, 0.98), 
            'eps': 1e-6
        },
        {
            'params': criterion.parameters(), 
            'lr': learning_rate * 0.1,
            'weight_decay': 0
        }
    ])

    scaler = GradScaler("cuda") if use_amp and device == 'cuda' else None
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3, 
        min_lr=1e-6
    ) if use_scheduler else None

    best_val_loss = float('inf')
    start_epoch = 0
    
    # Check for existing best model to know what we need to beat
    best_model_path = checkpoint_dir / 'best_model.pth'
    if best_model_path.exists():
        best_checkpoint = torch.load(best_model_path, map_location=device)
        existing_best_loss = best_checkpoint.get('val_loss', float('inf'))
        existing_best_epoch = best_checkpoint.get('epoch', -1) + 1  # Convert to 1-based
        print(f"\n📊 Existing best model: val_loss={existing_best_loss:.4f} from epoch {existing_best_epoch}")
        best_val_loss = existing_best_loss
    
    # Load specific epoch if requested
    checkpoint_to_load = None
    
    if resume_epoch is not None:
        # Look for specific epoch checkpoint
        specific_checkpoint = checkpoint_dir / f'checkpoint_epoch_{resume_epoch}.pth'
        if specific_checkpoint.exists():
            checkpoint_to_load = specific_checkpoint
            print(f"\n📂 Loading checkpoint from epoch {resume_epoch}: {specific_checkpoint}")
        else:
            print(f"\n⚠️ WARNING: No checkpoint found for epoch {resume_epoch}")
            print(f"  Looking for: {specific_checkpoint}")
            
            # List available checkpoints
            epoch_checkpoints = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
            if epoch_checkpoints:
                print("\n  Available epoch checkpoints:")
                for cp in epoch_checkpoints:
                    epoch_num = cp.stem.replace('checkpoint_epoch_', '')
                    print(f"     - Epoch {epoch_num}")
            
            response = input("\n. Continue without loading checkpoint? (y/n): ")
            if response.lower() != 'y':
                exit(0)
    
    else:
        latest_checkpoint = checkpoint_dir / 'latest_checkpoint.pth'
        if latest_checkpoint.exists():
            checkpoint_to_load = latest_checkpoint
            print(f"\n📂 Loading LATEST checkpoint from: {latest_checkpoint}")
    
    if checkpoint_to_load:
        checkpoint = torch.load(checkpoint_to_load, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

        if 'criterion_state_dict' in checkpoint:
            criterion.load_state_dict(checkpoint['criterion_state_dict'])
            print(f"✓ Loaded loss weights from checkpoint")
        else:
            print(f"⚠️ No loss weights in checkpoint - starting fresh")
        
        # Handle force_resume - reset learning rate and scheduler
        if force_resume:
            print(f"\n⚙️ FORCE MODE: Resetting learning rate to {learning_rate:.6f}")
            # Reset optimizer learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            # Don't load scheduler state - let it start fresh
            if scheduler:
                print(f"  Scheduler reset - will reduce LR on plateau with patience={scheduler.patience}")
        else:
            # Normal mode - load scheduler state if available
            if scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"✓ Loaded scheduler state (current LR: {optimizer.param_groups[0]['lr']:.6f})")
        
        print(f"✓ Loaded checkpoint from epoch {start_epoch}")
        print(f"  Train loss at epoch {start_epoch}: {checkpoint.get('train_loss', 'N/A'):.4f}")
        print(f"  Val loss at epoch {start_epoch}: {checkpoint.get('val_loss', 'N/A'):.4f}")
        print(f"  Will resume training from epoch {start_epoch}")
        
        # Show what we need to beat
        if best_val_loss != float('inf'):
            print(f"\n🎯 Target: Beat validation loss of {best_val_loss:.4f}")
        else:
            print(f"\n🎯 No previous best model found - establishing new baseline")

    print("\n" + "="*60)
    print(f"Training from epoch {start_epoch + 1} to {num_epochs}")
    print(f"Batch size: {batch_size}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    print(f"Accumulation steps: {accumulation_steps}, Save every: {save_every} epochs")
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
            current_lr = optimizer.param_groups[0]['lr']
            print(f"LR: {current_lr:.6f}")
        
        # Monitor adaptive weights
        if hasattr(criterion, 'log_weight'):
            log_w = torch.sigmoid(criterion.log_weight).item()
            hf_w = torch.sigmoid(criterion.hf_weight).item()
            print(f"[Epoch {epoch+1}] Adaptive weights - Log: {log_w:.3f}, HF: {hf_w:.3f}")

        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(f"[Epoch {epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Time: {epoch_time:.1f}s | "
              f"Speed: {len(train_loader)/epoch_time:.1f} batch/s")
        
        # Enhanced checkpoint saving with scheduler state
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'model_config': {'in_channels': 1, 'out_channels': 1, 'features': [64, 128, 256, 512]},
            'optimizer_state_dict': optimizer.state_dict(),
            'criterion_state_dict': criterion.state_dict(), 
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }
        
        if scheduler:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        
        # Always save latest checkpoint
        torch.save(checkpoint_data, checkpoint_dir / 'latest_checkpoint.pth')
        
        # Save best model if this is the best so far
        if avg_val_loss < best_val_loss:
            improvement = best_val_loss - avg_val_loss
            best_val_loss = avg_val_loss
            torch.save(checkpoint_data, checkpoint_dir / 'best_model.pth')
            print(f"  ✓ New best model saved! val_loss: {avg_val_loss:.4f} (improved by {improvement:.4f})")
        
        # Save periodic checkpoints
        if (epoch + 1) % save_every == 0:
            torch.save(checkpoint_data, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth')
            print(f"  ✓ Checkpoint saved for epoch {epoch+1}")
            
        print("")

    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Final epoch: {num_epochs}")
    print("="*60)
    
    return model


if __name__ == '__main__':
    # Check if pre-computed specs exist
    spec_dir = Path('/scratch/egbueze.m/precomputed_specs_db120')
    
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
        num_epochs=100,
        batch_size=64,
        learning_rate=3e-4,
        device='cuda',
        checkpoint_dir='/scratch/egbueze.m/checkpoints_db120_residual',
        save_every=2,
        accumulation_steps=1,
        use_amp=True,
        resume_epoch=None,
        force_resume=False
    )
