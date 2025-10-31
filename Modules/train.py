import dataset
import unet
import torch
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader

def train_model(
    train_dataset,
    val_dataset,
    num_epochs=50,
    batch_size=16,
    learning_rate=1e-3,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    checkpoint_dir='checkpoints',
    save_every=5
):
    """
    Train the U-Net model with checkpointing
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        device: 'cuda' or 'cpu'
        checkpoint_dir: Directory to save checkpoints
        save_every: Save checkpoint every N epochs
    """
    
    # Create checkpoint directory
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Initialize model
    model = unet.UNet(in_channels=1, out_channels=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # For tracking best model
    best_val_loss = float('inf')
    
    print(f"Training on {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Total epochs: {num_epochs}")
    print("=" * 60)
    
    # Training loop
    for epoch in range(num_epochs):
        # ========== Training Phase ==========
        model.train()
        train_loss = 0
        
        for batch_idx, (reverb_spec, clean_spec) in enumerate(train_loader):
            # Move to device
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
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        
        # ========== Validation Phase ==========
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
        
        # Print epoch summary
        print("=" * 60)
        print(f"Epoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        
        # ========== Save Best Model ==========
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_dir / 'best_model.pth')
            print(f"  ✓ Saved best model (val_loss: {avg_val_loss:.4f})")
        
        # ========== Save Periodic Checkpoint ==========
        if (epoch + 1) % save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth')
            print(f"  ✓ Saved checkpoint at epoch {epoch+1}")
        
        # ========== Save Latest Checkpoint ==========
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, checkpoint_dir / 'latest_checkpoint.pth')
        
        print("=" * 60)
    
    # ========== Save Final Model ==========
    torch.save(model.state_dict(), checkpoint_dir / 'final_model.pth')
    print("\n✓ Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved in: {checkpoint_dir}")
    
    return model


if __name__ == '__main__':
    # Setup paths
    data_dir = Path('./dereverb_dataset')
    
    # Create datasets (80/20 train/val split)
    all_files = sorted(list((data_dir / 'reverb').glob('*.flac')))
    split_idx = int(0.8 * len(all_files))
    
    train_dataset = dataset.DereverbDataset(
        reverb_dir=data_dir / 'reverb',
        clean_dir=data_dir / 'clean'
    )
    
    from torch.utils.data import random_split
    
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible split
    )
    
    # Train model
    model = train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=50,
        batch_size=16,
        learning_rate=1e-3,
        checkpoint_dir='checkpoints',
        save_every=5
    )
            