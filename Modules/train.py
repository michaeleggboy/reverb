from pathlib import Path
import time
import dataset
import unet
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast


def train_model(
    train_dataset,
    val_dataset,
    num_epochs=40,
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
    Train the U-Net model with optimizations

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate for optimizer
        device: 'cuda' or 'cpu'
        checkpoint_dir: Directory to save checkpoints
        save_every: Save checkpoint every N epochs
        resume_from: Path to checkpoint file to resume from (optional)
        use_scheduler: Whether to use learning rate scheduler
        accumulation_steps: Gradient accumulation steps (effective batch = batch_size * accumulation_steps)
        use_amp: Use mixed precision training (AMP)
    """

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

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

    model = unet.UNet(in_channels=1, out_channels=1).to(device)
    
    if hasattr(torch, 'compile'):
        model = torch.compile(model)
        print("✓ Model compiled with torch.compile()")
    
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

    if resume_from:
        resume_path = Path(resume_from)
        if resume_path.exists():
            print(f"📂 Loading checkpoint from: {resume_path}")
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
            print(f"  Continuing from epoch {start_epoch}")
        else:
            print(f"⚠️  Checkpoint not found: {resume_path}")
            print("   Starting training from scratch")

    elif (checkpoint_dir / 'latest_checkpoint.pth').exists():
        resume_path = checkpoint_dir / 'latest_checkpoint.pth'
        print("📂 Found latest checkpoint, auto-resuming...")
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
        print(f"  Continuing from epoch {start_epoch}")

    print(f"\nTraining on {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Batch size: {batch_size}")
    if accumulation_steps > 1:
        print(f"Gradient accumulation: {accumulation_steps} (effective batch: {batch_size * accumulation_steps})")
    print(f"Starting epoch: {start_epoch}")
    print(f"Target epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    if use_scheduler:
        print("LR Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)")
    print("=" * 60)

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()

        model.train()
        train_loss = 0

        for batch_idx, (reverb_spec, clean_spec) in enumerate(train_loader):
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

            if (batch_idx + 1) % 100 == 0:
                current_lr = scheduler.get_last_lr()[0] if scheduler else learning_rate
                print(f"Epoch [{epoch+1}/{num_epochs}] "
                      f"Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item()*accumulation_steps:.4f} "
                      f"LR: {current_lr:.6f}")

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for reverb_spec, clean_spec in val_loader:
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

        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - epoch_start

        if scheduler is not None:
            scheduler.step(avg_val_loss)
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = learning_rate

        print("=" * 60)
        print(f"Epoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Time: {epoch_time:.1f}s ({epoch_time/60:.1f} min)")

        if epoch < num_epochs - 1:
            remaining = (num_epochs - epoch - 1) * epoch_time
            print(f"  Est. remaining: {remaining/60:.1f} min ({remaining/3600:.1f} hours)")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }

            if scheduler is not None:
                checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
            
            if scaler is not None:
                checkpoint_data['scaler_state_dict'] = scaler.state_dict()

            torch.save(checkpoint_data, checkpoint_dir / 'best_model.pth')
            print(f"  ✓ Saved best model (val_loss: {avg_val_loss:.4f})")

        if (epoch + 1) % save_every == 0:
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }

            if scheduler is not None:
                checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
            
            if scaler is not None:
                checkpoint_data['scaler_state_dict'] = scaler.state_dict()

            torch.save(checkpoint_data, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth')
            print(f"  ✓ Saved checkpoint at epoch {epoch+1}")

        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }

        if scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        
        if scaler is not None:
            checkpoint_data['scaler_state_dict'] = scaler.state_dict()

        torch.save(checkpoint_data, checkpoint_dir / 'latest_checkpoint.pth')

        print("=" * 60)

    torch.save(model.state_dict(), checkpoint_dir / 'final_model.pth')
    print("\n✓ Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved in: {checkpoint_dir}")

    return model


if __name__ == '__main__':
    data_dir = Path('./drive/MyDrive/dereverb_dataset')
    checkpoint_dir = Path('./drive/MyDrive/checkpoints_85k')

    full_dataset = dataset.DereverbDataset(
        reverb_dir=data_dir / 'reverb',
        clean_dir=data_dir / 'clean'
    )

    from torch.utils.data import random_split

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    model = train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=40,
        batch_size=32,
        learning_rate=1e-3,
        device='cuda',
        checkpoint_dir=checkpoint_dir,
        save_every=5,
        resume_from=None,
        use_scheduler=True,
        accumulation_steps=2,
        use_amp=True,
    )

    print("\n🎉 Training complete!")
    print(f"Best model saved at: {checkpoint_dir / 'best_model.pth'}")