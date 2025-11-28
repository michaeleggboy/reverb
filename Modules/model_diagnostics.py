from precomputed_dataset import PrecomputedDataset
from unet import UNet
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt


def diagnose_model_output(model, test_loader, device='cuda', verbose=True):
    """
    Comprehensive diagnosis for direct-output dereverberation model.
    Analyzes what the model is learning to do with spectrograms.
    """
    
    model.eval()
    diagnostics = {
        'input_ranges': [],
        'output_ranges': [],
        'target_ranges': [],
        'changes': [],
        'preservation_ratios': [],
        'freq_responses': [],
        'energy_ratios': [],
        'implicit_masks': []  # What mask would produce this output?
    }
    
    with torch.no_grad():
        for batch_idx, (reverb, clean) in enumerate(test_loader):
            if batch_idx >= 5:  # Analyze 5 batches for statistics
                break
                
            reverb = reverb.to(device)
            clean = clean.to(device)
            pred = model(reverb)
            
            # 1. Dynamic range analysis
            in_range = (reverb.min().item(), reverb.max().item())
            out_range = (pred.min().item(), pred.max().item())
            target_range = (clean.min().item(), clean.max().item())
            
            diagnostics['input_ranges'].append(in_range)
            diagnostics['output_ranges'].append(out_range)
            diagnostics['target_ranges'].append(target_range)
            
            # 2. Change magnitude
            change = torch.abs(pred - reverb).mean().item()
            diagnostics['changes'].append(change)
            
            # 3. Implicit mask calculation
            # If model was using masking, what mask would create this output?
            # Implicit_mask = pred / (reverb + eps)
            eps = 1e-7
            implicit_mask = pred / (reverb + eps)
            implicit_mask = torch.clamp(implicit_mask, 0, 2)  # Cap at 2x amplification
            diagnostics['implicit_masks'].append({
                'mean': implicit_mask.mean().item(),
                'median': implicit_mask.median().item(),
                'std': implicit_mask.std().item()
            })
            
            # 4. Preservation ratio (how much of input is preserved)
            preservation = (pred.sum() / (reverb.sum() + eps)).item()
            diagnostics['preservation_ratios'].append(preservation)
            
            # 5. Frequency-specific analysis
            # Assuming shape [B, C, F=256, T]
            freq_bins = pred.shape[-2]
            low_cutoff = freq_bins // 4    # Bottom 25%
            high_cutoff = 3 * freq_bins // 4  # Top 25%
            
            # Input energy by frequency band
            low_in = reverb[:, :, :low_cutoff, :].mean().item()
            mid_in = reverb[:, :, low_cutoff:high_cutoff, :].mean().item()
            high_in = reverb[:, :, high_cutoff:, :].mean().item()
            
            # Output energy by frequency band
            low_out = pred[:, :, :low_cutoff, :].mean().item()
            mid_out = pred[:, :, low_cutoff:high_cutoff, :].mean().item()
            high_out = pred[:, :, high_cutoff:, :].mean().item()
            
            # Target energy for comparison
            low_target = clean[:, :, :low_cutoff, :].mean().item()
            mid_target = clean[:, :, low_cutoff:high_cutoff, :].mean().item()
            high_target = clean[:, :, high_cutoff:, :].mean().item()
            
            diagnostics['freq_responses'].append({
                'low_ratio': low_out / (low_in + eps),
                'mid_ratio': mid_out / (mid_in + eps),
                'high_ratio': high_out / (high_in + eps),
                'low_target_ratio': low_target / (low_in + eps),
                'mid_target_ratio': mid_target / (mid_in + eps),
                'high_target_ratio': high_target / (high_in + eps)
            })
            
            # 6. Total energy preservation
            energy_ratio = (pred.pow(2).sum() / (reverb.pow(2).sum() + eps)).item()
            diagnostics['energy_ratios'].append(energy_ratio)
    
    # Calculate statistics
    avg_change = np.mean(diagnostics['changes'])
    avg_preservation = np.mean(diagnostics['preservation_ratios'])
    avg_energy_ratio = np.mean(diagnostics['energy_ratios'])
    
    # Frequency response averages
    freq_stats = {}
    for band in ['low_ratio', 'mid_ratio', 'high_ratio']:
        freq_stats[band] = np.mean([f[band] for f in diagnostics['freq_responses']])
    
    # Implicit mask statistics
    mask_stats = {
        'mean': np.mean([m['mean'] for m in diagnostics['implicit_masks']]),
        'median': np.mean([m['median'] for m in diagnostics['implicit_masks']]),
        'std': np.mean([m['std'] for m in diagnostics['implicit_masks']])
    }
    
    if verbose:
        print("="*60)
        print("MODEL BEHAVIOR DIAGNOSIS (Direct Output)")
        print("="*60)
        
        print(f"\n📊 RANGES:")
        avg_in_min = np.mean([r[0] for r in diagnostics['input_ranges']])
        avg_in_max = np.mean([r[1] for r in diagnostics['input_ranges']])
        avg_out_min = np.mean([r[0] for r in diagnostics['output_ranges']])
        avg_out_max = np.mean([r[1] for r in diagnostics['output_ranges']])
        
        print(f"  Input:  [{avg_in_min:.3f}, {avg_in_max:.3f}]")
        print(f"  Output: [{avg_out_min:.3f}, {avg_out_max:.3f}]")
        print(f"  Expected: [0.0, 1.0]")
        
        print(f"\n📊 CHANGES:")
        print(f"  Avg change from input: {avg_change:.4f}")
        print(f"  Preservation ratio: {avg_preservation:.2%}")
        print(f"  Energy ratio: {avg_energy_ratio:.2%}")
        
        print(f"\n📊 FREQUENCY RESPONSE:")
        print(f"  Low freq:  {freq_stats['low_ratio']:.2%}")
        print(f"  Mid freq:  {freq_stats['mid_ratio']:.2%}")
        print(f"  High freq: {freq_stats['high_ratio']:.2%}")
        
        print(f"\n📊 IMPLICIT MASK ANALYSIS:")
        print(f"  What mask would create this output?")
        print(f"  Mean: {mask_stats['mean']:.3f} (1.0 = no change)")
        print(f"  Median: {mask_stats['median']:.3f}")
        print(f"  Std: {mask_stats['std']:.3f} (variation in masking)")
        
        print(f"\n⚠️  POTENTIAL ISSUES:")
        
        # Diagnostic interpretations
        if avg_change < 0.01:
            print("  🔴 Model barely changing input (might be copying)")
        elif avg_change > 0.5:
            print("  🔴 Model changing too aggressively")
        else:
            print("  ✅ Reasonable change magnitude")
        
        if freq_stats['high_ratio'] < 0.5:
            print("  🟡 Losing high frequencies (muffled output)")
        elif freq_stats['high_ratio'] > 1.2:
            print("  🟡 Amplifying high frequencies (harsh/metallic)")
        
        if freq_stats['low_ratio'] < 0.7:
            print("  🟡 Losing low frequencies (thin output)")
        elif freq_stats['low_ratio'] > 1.2:
            print("  🟡 Amplifying low frequencies (boomy)")
        
        if mask_stats['mean'] > 0.95:
            print("  🟡 Very conservative (not removing much reverb)")
        elif mask_stats['mean'] < 0.5:
            print("  🟡 Very aggressive (might remove speech)")
        
        if mask_stats['std'] < 0.05:
            print("  🟡 Uniform processing (not adapting to content)")
        
        if avg_out_min < -0.1 or avg_out_max > 1.1:
            print("  🔴 Outputs outside valid range!")
        
        print("="*60)
    
    return diagnostics


def visualize_model_behavior(model, test_loader, device='cuda', save_path=None):
    """
    Create visual diagnosis of model behavior on a sample
    """
    model.eval()
    
    with torch.no_grad():
        # Get one batch
        reverb, clean = next(iter(test_loader))
        reverb = reverb.to(device)
        clean = clean.to(device)
        pred = model(reverb)
        
        # Take first sample
        reverb_np = reverb[0, 0].cpu().numpy()
        clean_np = clean[0, 0].cpu().numpy()
        pred_np = pred[0, 0].cpu().numpy()
        
        # Calculate difference maps
        reverb_clean_diff = reverb_np - clean_np  # What reverb added
        pred_clean_diff = pred_np - clean_np      # Model error
        reverb_pred_diff = reverb_np - pred_np    # What model removed
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Row 1: Spectrograms
        im1 = axes[0, 0].imshow(reverb_np, aspect='auto', origin='lower', cmap='viridis')
        axes[0, 0].set_title('Input (Reverberant)')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Frequency')
        plt.colorbar(im1, ax=axes[0, 0])
        
        im2 = axes[0, 1].imshow(pred_np, aspect='auto', origin='lower', cmap='viridis')
        axes[0, 1].set_title('Model Output')
        axes[0, 1].set_xlabel('Time')
        plt.colorbar(im2, ax=axes[0, 1])
        
        im3 = axes[0, 2].imshow(clean_np, aspect='auto', origin='lower', cmap='viridis')
        axes[0, 2].set_title('Target (Clean)')
        axes[0, 2].set_xlabel('Time')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Row 2: Difference maps
        im4 = axes[1, 0].imshow(reverb_pred_diff, aspect='auto', origin='lower', cmap='RdBu', vmin=-0.5, vmax=0.5)
        axes[1, 0].set_title('What Model Removed')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Frequency')
        plt.colorbar(im4, ax=axes[1, 0])
        
        im5 = axes[1, 1].imshow(pred_clean_diff, aspect='auto', origin='lower', cmap='RdBu', vmin=-0.5, vmax=0.5)
        axes[1, 1].set_title('Model Error (Pred - Target)')
        axes[1, 1].set_xlabel('Time')
        plt.colorbar(im5, ax=axes[1, 1])
        
        # Frequency response plot
        freq_response_in = reverb_np.mean(axis=1)
        freq_response_out = pred_np.mean(axis=1)
        freq_response_target = clean_np.mean(axis=1)
        
        axes[1, 2].plot(freq_response_in, label='Input', alpha=0.7)
        axes[1, 2].plot(freq_response_out, label='Output', alpha=0.7)
        axes[1, 2].plot(freq_response_target, label='Target', alpha=0.7, linestyle='--')
        axes[1, 2].set_title('Frequency Response (Avg)')
        axes[1, 2].set_xlabel('Frequency Bin')
        axes[1, 2].set_ylabel('Magnitude')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
        
    return fig


if __name__ == '__main__':
    # Load trained model
    model = UNet(in_channels=1, out_channels=1).to('cuda')
    checkpoint = torch.load('/scratch/egbueze.m/checkpoints_normalized/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load validation dataset
    val_dataset = PrecomputedDataset('/scratch/egbueze.m/precomputed_specs_normalized')
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Run diagnosis
    diagnostics = diagnose_model_output(model, val_loader)

    # Create visualization
    visualize_model_behavior(model, val_loader, save_path='model_analysis.png')
