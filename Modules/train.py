from pathlib import Path
import time
import dataset
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
    """
    Train the U-Net model with optimizations and monitoring
    """
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
        num_workers=2,
        pin_memory=True if device == 'cuda' else False,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device == 'cuda' else False,
        persistent_workers=True,
        prefetch_factor=2,
    )

    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")

    # Initialize model
    print("\nInitializing model...")
    model = unet.UNet(in_channels=1, out_channels=1).to(device)

    # Compile if available
    if hasattr(torch, 'compile'):
        model = torch.compile(model, mode="reduce-overhead")

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )

    scaler = GradScaler("cuda") if use_amp and device == 'cuda' else None
    if scaler:
        print("✓ Mixed precision training enabled (AMP)")

    scheduler = None
    if use_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        print("✓ Learning rate scheduler enabled")

    best_val_loss = float('inf')
    start_epoch = 0

    # Handle checkpoint loading
    if resume_from:
        resume_path = Path(resume_from)
        if resume_path.exists():
            print(f"\n📂 Loading checkpoint from: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if use_scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            if scaler and 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])

            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('val_loss', float('inf'))

            print(f"✓ Resumed from epoch {checkpoint['epoch']}")
            print(f"  Previous train loss: {checkpoint.get('train_loss', 'N/A'):.4f}")
            print(f"  Previous val loss: {checkpoint.get('val_loss', 'N/A'):.4f}")

    elif (checkpoint_dir / 'latest_checkpoint.pth').exists():
        resume_path = checkpoint_dir / 'latest_checkpoint.pth'
        print("\n📂 Found latest checkpoint, auto-resuming...")
        checkpoint = torch.load(resume_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if use_scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))

        print(f"✓ Auto-resumed from epoch {checkpoint['epoch']}")

    # Print training configuration
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Batches per epoch: {len(train_loader)}")
    if accumulation_steps > 1:
        print(f"Gradient accumulation: {accumulation_steps} (effective batch: {batch_size * accumulation_steps})")
    print(f"Starting epoch: {start_epoch}")
    print(f"Target epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    if use_scheduler:
        print("LR Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)")
    print("="*60 + "\n")

    # Enable TF32 for A100/L4
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }

    # Main training loop
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()

        # Training phase
        model.train()
        train_loss = 0
        batch_times = []

        print(f"\n[EPOCH {epoch+1}/{num_epochs}]")
        print("-" * 40)

        # Progress bar for training
        train_pbar = tqdm(train_loader, desc="Training", leave=False)

        for batch_idx, (reverb_spec, clean_spec) in enumerate(train_pbar):
            batch_start = time.time()

            # First batch notification
            if batch_idx == 0:
                print("  ✓ First batch loaded successfully")

            reverb_spec = reverb_spec.to(device, non_blocking=True)
            clean_spec = clean_spec.to(device, non_blocking=True)

            if scaler:
                with autocast("cuda"):
                    pred_spec = model(reverb_spec)
                    loss = criterion(pred_spec, clean_spec)
                    loss = loss / accumulation_steps

                scaler.scale(loss).backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                pred_spec = model(reverb_spec)
                loss = criterion(pred_spec, clean_spec)
                loss = loss / accumulation_steps

                loss.backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * accumulation_steps

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            # Update progress bar
            current_lr = scheduler.get_last_lr()[0] if scheduler else learning_rate
            train_pbar.set_postfix({
                'loss': f'{loss.item()*accumulation_steps:.4f}',
                'lr': f'{current_lr:.6f}',
                'speed': f'{1/np.mean(batch_times[-10:]) if batch_times else 0:.1f} b/s'
            })

            # Detailed progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                avg_batch_time = np.mean(batch_times[-100:])
                batches_remaining = len(train_loader) - batch_idx - 1
                eta = batches_remaining * avg_batch_time

                print(f"  Progress: [{batch_idx+1}/{len(train_loader)}] "
                      f"({100*(batch_idx+1)/len(train_loader):.1f}%) "
                      f"| Loss: {loss.item()*accumulation_steps:.4f} "
                      f"| Speed: {1/avg_batch_time:.1f} batch/s "
                      f"| ETA: {eta/60:.1f} min")

        avg_train_loss = train_loss / len(train_loader)
        train_pbar.close()

        print("\n  Validating...")
        model.eval()
        val_loss = 0

        val_pbar = tqdm(val_loader, desc="Validation", leave=False)

        with torch.no_grad():
            for reverb_spec, clean_spec in val_pbar:
                reverb_spec = reverb_spec.to(device, non_blocking=True)
                clean_spec = clean_spec.to(device, non_blocking=True)

                if scaler:
                    with autocast("cuda"):
                        pred_spec = model(reverb_spec)
                        loss = criterion(pred_spec, clean_spec)
                else:
                    pred_spec = model(reverb_spec)
                    loss = criterion(pred_spec, clean_spec)

                val_loss += loss.item()

                val_pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})

        avg_val_loss = val_loss / len(val_loader)
        val_pbar.close()

        epoch_time = time.time() - epoch_start

        # Update scheduler
        if scheduler is not None:
            scheduler.step(avg_val_loss)
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = learning_rate

        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['learning_rates'].append(current_lr)

        # Print epoch summary
        print("\n" + "="*60)
        print(f"EPOCH {epoch+1}/{num_epochs} SUMMARY")
        print("="*60)
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Time: {epoch_time:.1f}s ({epoch_time/60:.1f} min)")
        print(f"  Speed: {len(train_loader)/epoch_time:.1f} batches/sec")

        # GPU memory usage
        if device == 'cuda':
            mem_used = torch.cuda.memory_allocated() / 1e9
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU Memory: {mem_used:.1f}/{mem_total:.1f} GB ({100*mem_used/mem_total:.0f}%)")

        if epoch < num_epochs - 1:
            remaining = (num_epochs - epoch - 1) * epoch_time
            print(f"  Est. remaining: {remaining/60:.1f} min ({remaining/3600:.1f} hours)")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'history': history
            }

            if scheduler is not None:
                checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()

            if scaler is not None:
                checkpoint_data['scaler_state_dict'] = scaler.state_dict()

            torch.save(checkpoint_data, checkpoint_dir / 'best_model.pth')
            print(f"  ✓ Saved best model (val_loss: {avg_val_loss:.4f})")

        # Periodic checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'history': history
            }

            if scheduler is not None:
                checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()

            if scaler is not None:
                checkpoint_data['scaler_state_dict'] = scaler.state_dict()

            torch.save(checkpoint_data, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth')
            print(f"  ✓ Saved checkpoint at epoch {epoch+1}")

        # Always save latest
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'history': history
        }

        if scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()

        if scaler is not None:
            checkpoint_data['scaler_state_dict'] = scaler.state_dict()

        torch.save(checkpoint_data, checkpoint_dir / 'latest_checkpoint.pth')

        print("="*60)

    # Final save
    torch.save(model.state_dict(), checkpoint_dir / 'final_model.pth')

    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"Checkpoints saved in: {checkpoint_dir}")
    print("="*60)

    return model, history


if __name__ == '__main__':
    data_dir = Path('/content/drive/MyDrive/dereverb_dataset')
    checkpoint_dir = Path('/content/drive/MyDrive/checkpoints_85k')

    print("Loading dataset...")
    full_dataset = dataset.DereverbDataset(
        reverb_dir=data_dir / 'reverb',
        clean_dir=data_dir / 'clean'
    )
    print(f"✓ Loaded {len(full_dataset)} pairs")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    model, history = train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=25,
        batch_size=64,
        learning_rate=1e-3,
        device='cuda',
        checkpoint_dir=checkpoint_dir,
        save_every=5,
        resume_from=None,
        use_scheduler=True,
        accumulation_steps=1,
        use_amp=True,
    )

    print("\n🎉 All done!")
