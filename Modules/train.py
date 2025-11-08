import dataset
import unet
import torch
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time


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
    use_scheduler=True
):
    """
    Train the U-Net model with checkpointing and learning rate scheduling
    
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
    """
    
    # Create checkpoint directory
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Initialize model
    model = unet.UNet(in_channels=1, out_channels=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # ===== LEARNING RATE SCHEDULER =====
    scheduler = None
    if use_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',           # Minimize validation loss
            factor=0.5,           # Reduce LR by half
            patience=5,           # Wait 5 epochs before reducing
            verbose=True,         # Print when LR changes
            min_lr=1e-6          # Don't go below this
        )
        print("✓ Learning rate scheduler enabled")
    
    # For tracking best model and starting epoch
    best_val_loss = float('inf')
    start_epoch = 0
    
    # ===== RESUME FROM CHECKPOINT =====
    if resume_from:
        resume_path = Path(resume_from)
        if resume_path.exists():
            print(f"📂 Loading checkpoint from: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            
            # Restore model and optimizer
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Restore scheduler if it exists
            if use_scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Restore epoch and losses
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            
            print(f"✓ Resumed from epoch {checkpoint['epoch']}")
            print(f"  Previous train loss: {checkpoint.get('train_loss', 'N/A'):.4f}")
            print(f"  Previous val loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
            print(f"  Continuing from epoch {start_epoch}")
        else:
            print(f"⚠️  Checkpoint not found: {resume_path}")
            print("   Starting training from scratch")
    
    # ===== AUTO-RESUME FROM LATEST =====
    elif (checkpoint_dir / 'latest_checkpoint.pth').exists():
        resume_path = checkpoint_dir / 'latest_checkpoint.pth'
        print("📂 Found latest checkpoint, auto-resuming...")
        checkpoint = torch.load(resume_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if use_scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        
        print(f"✓ Auto-resumed from epoch {checkpoint['epoch']}")
        print(f"  Continuing from epoch {start_epoch}")
    
    print(f"\nTraining on {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Starting epoch: {start_epoch}")
    print(f"Target epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    if use_scheduler:
        print("LR Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)")
    print("=" * 60)
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        
        # ===== TRAINING PHASE =====
        model.train()
        train_loss = 0
        
        for batch_idx, (reverb_spec, clean_spec) in enumerate(train_loader):
            reverb_spec = reverb_spec.to(device)
            clean_spec = clean_spec.to(device)
            
            # Forward pass
            pred_spec = model(reverb_spec)
            loss = criterion(pred_spec, clean_spec)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Print progress
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        
        # ===== VALIDATION PHASE =====
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for reverb_spec, clean_spec in val_loader:
                reverb_spec = reverb_spec.to(device)
                clean_spec = clean_spec.to(device)
                
                pred_spec = model(reverb_spec)
                loss = criterion(pred_spec, clean_spec)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - epoch_start
        
        # ===== LEARNING RATE SCHEDULER =====
        if scheduler is not None:
            # Update learning rate based on validation loss
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = learning_rate
        
        # Print epoch summary
        print("=" * 60)
        print(f"Epoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Time: {epoch_time:.1f}s ({epoch_time/60:.1f} min)")
        
        # Estimate remaining time
        if epoch < num_epochs - 1:
            remaining = (num_epochs - epoch - 1) * epoch_time
            print(f"  Est. remaining: {remaining/60:.1f} min ({remaining/3600:.1f} hours)")
        
        # ===== SAVE BEST MODEL =====
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }
            
            # Save scheduler state if using scheduler
            if scheduler is not None:
                checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
            
            torch.save(checkpoint_data, checkpoint_dir / 'best_model.pth')
            print(f"  ✓ Saved best model (val_loss: {avg_val_loss:.4f})")
        
        # ===== SAVE PERIODIC CHECKPOINT =====
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
            
            torch.save(checkpoint_data, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth')
            print(f"  ✓ Saved checkpoint at epoch {epoch+1}")
        
        # ===== SAVE LATEST CHECKPOINT =====
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }
        
        if scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        
        torch.save(checkpoint_data, checkpoint_dir / 'latest_checkpoint.pth')
        
        print("=" * 60)
    
    # ===== SAVE FINAL MODEL =====
    torch.save(model.state_dict(), checkpoint_dir / 'final_model.pth')
    print("\n✓ Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved in: {checkpoint_dir}")
    
    return model


if __name__ == '__main__':
    # Setup paths
    data_dir = Path('./drive/MyDrive/dereverb_dataset')
    checkpoint_dir = Path('./drive/MyDrive/checkpoints_85k')
    
    # Create dataset
    full_dataset = dataset.DereverbDataset(
        reverb_dir=data_dir / 'reverb',
        clean_dir=data_dir / 'clean'
    )
    
    # Split into train/val (80/20)
    from torch.utils.data import random_split
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible split
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
        use_scheduler=True
    )
    
    print("\n🎉 Training complete!")
    print(f"Best model saved at: {checkpoint_dir / 'best_model.pth'}")
