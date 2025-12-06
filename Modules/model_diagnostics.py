from precomputed_dataset import PrecomputedDataset
from unet import UNet
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def check_raw_audio_difference(data_dir, num_samples=20):
    """
    Check how different reverb and clean audio files actually are
    before any spectrogram processing.
    """
    reverb_dir = Path(data_dir) / 'reverb'
    clean_dir = Path(data_dir) / 'clean'
    
    reverb_files = sorted(list(reverb_dir.glob('*.flac')))[:num_samples]
    
    differences = []
    max_diffs = []
    energy_ratios = []
    
    print("="*60)
    print("RAW AUDIO ANALYSIS (Before Spectrogram)")
    print("="*60)
    
    for reverb_file in reverb_files:
        clean_file = clean_dir / reverb_file.name
        
        if not clean_file.exists():
            continue
            
        reverb_audio, sr = torchaudio.load(str(reverb_file))
        clean_audio, _ = torchaudio.load(str(clean_file))
        
        # Ensure same length
        min_len = min(reverb_audio.shape[-1], clean_audio.shape[-1])
        reverb_audio = reverb_audio[..., :min_len]
        clean_audio = clean_audio[..., :min_len]
        
        # Calculate differences
        diff = torch.abs(reverb_audio - clean_audio)
        l1_diff = diff.mean().item()
        max_diff = diff.max().item()
        
        # Energy comparison
        reverb_energy = (reverb_audio ** 2).sum().item()
        clean_energy = (clean_audio ** 2).sum().item()
        energy_ratio = reverb_energy / (clean_energy + 1e-7)
        
        differences.append(l1_diff)
        max_diffs.append(max_diff)
        energy_ratios.append(energy_ratio)
    
    print(f"\n📊 AUDIO WAVEFORM DIFFERENCES ({len(differences)} samples):")
    print(f"  Mean L1 diff:     {np.mean(differences):.6f}")
    print(f"  Std L1 diff:      {np.std(differences):.6f}")
    print(f"  Max L1 diff:      {np.max(differences):.6f}")
    print(f"  Min L1 diff:      {np.min(differences):.6f}")
    print(f"\n  Mean max diff:    {np.mean(max_diffs):.4f}")
    print(f"  Mean energy ratio: {np.mean(energy_ratios):.4f} (1.0 = same energy)")
    
    # Interpretation
    print(f"\n⚠️  INTERPRETATION:")
    if np.mean(differences) < 0.01:
        print("  🔴 Audio files are VERY similar - reverb effect is subtle")
        print("     Consider regenerating with stronger RT60 (0.8-2.0s)")
    elif np.mean(differences) < 0.05:
        print("  🟡 Audio files have mild differences - reverb is moderate")
    else:
        print("  ✅ Audio files have clear differences - reverb is noticeable")
    
    if np.mean(energy_ratios) > 1.3:
        print(f"  ℹ️  Reverb adds {(np.mean(energy_ratios)-1)*100:.1f}% energy (expected)")
    
    print("="*60)
    
    return {
        'l1_diffs': differences,
        'max_diffs': max_diffs,
        'energy_ratios': energy_ratios
    }


def check_spectrogram_difference(spec_dir, num_samples=50):
    """
    Check differences in the precomputed spectrograms.
    """
    spec_dir = Path(spec_dir)
    spec_files = sorted(list(spec_dir.glob('spec_*.pt')))[:num_samples]
    
    differences = []
    max_diffs = []
    
    print("\n" + "="*60)
    print("SPECTROGRAM ANALYSIS (After Processing)")
    print("="*60)
    
    for spec_file in spec_files:
        data = torch.load(spec_file)
        reverb_spec = data['reverb']
        clean_spec = data['clean']
        
        diff = torch.abs(reverb_spec - clean_spec)
        l1_diff = diff.mean().item()
        max_diff = diff.max().item()
        
        differences.append(l1_diff)
        max_diffs.append(max_diff)
    
    print(f"\n📊 SPECTROGRAM DIFFERENCES ({len(differences)} samples):")
    print(f"  Mean L1 diff:     {np.mean(differences):.6f}")
    print(f"  Std L1 diff:      {np.std(differences):.6f}")
    print(f"  Max L1 diff:      {np.max(differences):.6f}")
    print(f"  Min L1 diff:      {np.min(differences):.6f}")
    print(f"\n  Mean max diff:    {np.mean(max_diffs):.4f}")
    
    # Compare to model's task difficulty
    print(f"\n⚠️  TASK DIFFICULTY:")
    if np.mean(differences) < 0.005:
        print("  🔴 Spectrograms nearly identical - model has almost nothing to learn")
        print("     This explains why model appears to 'copy' input")
    elif np.mean(differences) < 0.02:
        print("  🟡 Small differences - task is learnable but subtle")
    else:
        print("  ✅ Clear differences - meaningful dereverberation task")
    
    print("="*60)
    
    return {
        'l1_diffs': differences,
        'max_diffs': max_diffs
    }


def diagnose_model_output(model, test_loader, device='cuda', verbose=True):
    """
    Comprehensive diagnosis for direct-output dereverberation model.
    Supports both standard and mask-head UNet.
    """
    model.eval()
    has_mask = hasattr(model, 'mask_out')
    
    diagnostics = {
        'input_ranges': [],
        'output_ranges': [],
        'target_ranges': [],
        'changes': [],
        'preservation_ratios': [],
        'freq_responses': [],
        'energy_ratios': [],
        'implicit_masks': [],
        'identity_losses': [],
        'model_losses': [],
        'improvements': [],
        'mask_stats': [] if has_mask else None
    }
    
    with torch.no_grad():
        for batch_idx, (reverb, clean) in enumerate(test_loader):
            if batch_idx >= 5:
                break
            
            reverb = reverb.to(device)
            clean = clean.to(device)
            
            # Handle mask output if available
            if has_mask:
                pred, mask = model(reverb, return_mask=True)
                diagnostics['mask_stats'].append({
                    'mean': mask.mean().item(),
                    'std': mask.std().item(),
                    'min': mask.min().item(),
                    'max': mask.max().item(),
                    'zeros': (mask < 0.1).float().mean().item(),
                    'ones': (mask > 0.9).float().mean().item()
                })
            else:
                pred = model(reverb)
            
            # Loss comparison
            identity_l1 = F.l1_loss(reverb, clean).item()
            model_l1 = F.l1_loss(pred, clean).item()
            improvement = (identity_l1 - model_l1) / (identity_l1 + 1e-7) * 100
            
            diagnostics['identity_losses'].append(identity_l1)
            diagnostics['model_losses'].append(model_l1)
            diagnostics['improvements'].append(improvement)
            
            # Dynamic range
            diagnostics['input_ranges'].append((reverb.min().item(), reverb.max().item()))
            diagnostics['output_ranges'].append((pred.min().item(), pred.max().item()))
            diagnostics['target_ranges'].append((clean.min().item(), clean.max().item()))
            
            # Change magnitude
            change = torch.abs(pred - reverb).mean().item()
            diagnostics['changes'].append(change)
            
            # Implicit mask
            eps = 1e-7
            implicit_mask = pred / (reverb + eps)
            implicit_mask = torch.clamp(implicit_mask, 0, 2)
            diagnostics['implicit_masks'].append({
                'mean': implicit_mask.mean().item(),
                'median': implicit_mask.median().item(),
                'std': implicit_mask.std().item()
            })
            
            # Preservation ratio
            preservation = (pred.sum() / (reverb.sum() + eps)).item()
            diagnostics['preservation_ratios'].append(preservation)
            
            # Frequency analysis
            freq_bins = pred.shape[-2]
            low_cutoff = freq_bins // 4
            high_cutoff = 3 * freq_bins // 4
            
            low_in = reverb[:, :, :low_cutoff, :].mean().item()
            mid_in = reverb[:, :, low_cutoff:high_cutoff, :].mean().item()
            high_in = reverb[:, :, high_cutoff:, :].mean().item()
            
            low_out = pred[:, :, :low_cutoff, :].mean().item()
            mid_out = pred[:, :, low_cutoff:high_cutoff, :].mean().item()
            high_out = pred[:, :, high_cutoff:, :].mean().item()
            
            diagnostics['freq_responses'].append({
                'low_ratio': low_out / (low_in + eps),
                'mid_ratio': mid_out / (mid_in + eps),
                'high_ratio': high_out / (high_in + eps)
            })
            
            # Energy ratio
            energy_ratio = (pred.pow(2).sum() / (reverb.pow(2).sum() + eps)).item()
            diagnostics['energy_ratios'].append(energy_ratio)
    
    if verbose:
        print("\n" + "="*60)
        print("MODEL BEHAVIOR DIAGNOSIS")
        print("="*60)
        
        if has_mask:
            print(f"\n📊 MASK STATISTICS:")
            mask_mean = np.mean([m['mean'] for m in diagnostics['mask_stats']])
            mask_std = np.mean([m['std'] for m in diagnostics['mask_stats']])
            mask_zeros = np.mean([m['zeros'] for m in diagnostics['mask_stats']])
            mask_ones = np.mean([m['ones'] for m in diagnostics['mask_stats']])
            print(f"  Mean value:    {mask_mean:.3f}")
            print(f"  Std:           {mask_std:.3f}")
            print(f"  Near-zero (<0.1): {mask_zeros:.1%}")
            print(f"  Near-one (>0.9):  {mask_ones:.1%}")
        
        print(f"\n📊 LOSS COMPARISON:")
        print(f"  Identity (do nothing) L1: {np.mean(diagnostics['identity_losses']):.6f}")
        print(f"  Model output L1:          {np.mean(diagnostics['model_losses']):.6f}")
        print(f"  Improvement:              {np.mean(diagnostics['improvements']):.1f}%")
        
        print(f"\n📊 RANGES:")
        avg_in_min = np.mean([r[0] for r in diagnostics['input_ranges']])
        avg_in_max = np.mean([r[1] for r in diagnostics['input_ranges']])
        avg_out_min = np.mean([r[0] for r in diagnostics['output_ranges']])
        avg_out_max = np.mean([r[1] for r in diagnostics['output_ranges']])
        print(f"  Input:  [{avg_in_min:.3f}, {avg_in_max:.3f}]")
        print(f"  Output: [{avg_out_min:.3f}, {avg_out_max:.3f}]")
        
        print(f"\n📊 CHANGES:")
        avg_change = np.mean(diagnostics['changes'])
        print(f"  Avg change from input: {avg_change:.6f}")
        print(f"  Preservation ratio:    {np.mean(diagnostics['preservation_ratios']):.2%}")
        print(f"  Energy ratio:          {np.mean(diagnostics['energy_ratios']):.2%}")
        
        print(f"\n📊 FREQUENCY RESPONSE:")
        freq_stats = {k: np.mean([f[k] for f in diagnostics['freq_responses']]) 
                      for k in ['low_ratio', 'mid_ratio', 'high_ratio']}
        print(f"  Low freq:  {freq_stats['low_ratio']:.2%}")
        print(f"  Mid freq:  {freq_stats['mid_ratio']:.2%}")
        print(f"  High freq: {freq_stats['high_ratio']:.2%}")
        
        print(f"\n📊 IMPLICIT MASK:")
        mask_mean = np.mean([m['mean'] for m in diagnostics['implicit_masks']])
        mask_std = np.mean([m['std'] for m in diagnostics['implicit_masks']])
        print(f"  Mean: {mask_mean:.3f} (1.0 = no change)")
        print(f"  Std:  {mask_std:.3f}")
        
        print(f"\n⚠️  DIAGNOSIS:")
        identity_l1 = np.mean(diagnostics['identity_losses'])
        if identity_l1 < 0.005:
            print("  🔴 ROOT CAUSE: Input and target nearly identical (L1 < 0.005)")
            print("     The reverb effect is too subtle for meaningful learning")
            print("     → Regenerate data with stronger RT60 (0.8-2.0s)")
        elif avg_change < 0.001:
            print("  🔴 Model collapsed to identity despite learnable differences")
            print("     → Try residual learning or reduce skip connection strength")
        else:
            print("  ✅ Model is learning meaningful transformations")
        
        print("="*60)
    
    return diagnostics


def visualize_model_behavior(model, test_loader, device='cuda', save_path=None):
    """Create visual diagnosis of model behavior. Supports mask-head UNet."""
    model.eval()
    has_mask = hasattr(model, 'mask_out')
    
    with torch.no_grad():
        reverb, clean = next(iter(test_loader))
        reverb = reverb.to(device)
        clean = clean.to(device)
        
        if has_mask:
            pred, mask = model(reverb, return_mask=True)
            mask_np = mask[0, 0].cpu().numpy()
        else:
            pred = model(reverb)
            mask_np = None
        
        reverb_np = reverb[0, 0].cpu().numpy()
        clean_np = clean[0, 0].cpu().numpy()
        pred_np = pred[0, 0].cpu().numpy()
        
        reverb_clean_diff = reverb_np - clean_np
        pred_clean_diff = pred_np - clean_np
        reverb_pred_diff = reverb_np - pred_np
        
        # Create figure with extra row for mask if available
        n_rows = 3 if has_mask else 2
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
        
        im1 = axes[0, 0].imshow(reverb_np, aspect='auto', origin='lower', cmap='viridis')
        axes[0, 0].set_title('Input (Reverberant)')
        plt.colorbar(im1, ax=axes[0, 0])
        
        im2 = axes[0, 1].imshow(pred_np, aspect='auto', origin='lower', cmap='viridis')
        axes[0, 1].set_title('Model Output')
        plt.colorbar(im2, ax=axes[0, 1])
        
        im3 = axes[0, 2].imshow(clean_np, aspect='auto', origin='lower', cmap='viridis')
        axes[0, 2].set_title('Target (Clean)')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Use actual data range for difference maps
        vmax = max(abs(reverb_clean_diff).max(), 0.01)
        
        im4 = axes[1, 0].imshow(reverb_clean_diff, aspect='auto', origin='lower', 
                                 cmap='RdBu', vmin=-vmax, vmax=vmax)
        axes[1, 0].set_title(f'Reverb - Clean (max={vmax:.4f})')
        plt.colorbar(im4, ax=axes[1, 0])
        
        im5 = axes[1, 1].imshow(reverb_pred_diff, aspect='auto', origin='lower', 
                                 cmap='RdBu', vmin=-vmax, vmax=vmax)
        axes[1, 1].set_title('What Model Removed')
        plt.colorbar(im5, ax=axes[1, 1])
        
        im6 = axes[1, 2].imshow(pred_clean_diff, aspect='auto', origin='lower', 
                                 cmap='RdBu', vmin=-vmax, vmax=vmax)
        axes[1, 2].set_title('Model Error')
        plt.colorbar(im6, ax=axes[1, 2])
        
        # Mask visualization row
        if has_mask:
            im7 = axes[2, 0].imshow(mask_np, aspect='auto', origin='lower', cmap='gray', vmin=0, vmax=1)
            axes[2, 0].set_title(f'Learned Mask (mean={mask_np.mean():.3f})')
            plt.colorbar(im7, ax=axes[2, 0])
            
            # Mask histogram
            axes[2, 1].hist(mask_np.flatten(), bins=50, edgecolor='black', alpha=0.7)
            axes[2, 1].set_title('Mask Value Distribution')
            axes[2, 1].set_xlabel('Mask Value')
            axes[2, 1].set_ylabel('Count')
            axes[2, 1].axvline(x=0.5, color='r', linestyle='--', label='0.5 threshold')
            axes[2, 1].legend()
            
            # Mask vs clean energy correlation
            clean_energy = clean_np
            im8 = axes[2, 2].scatter(clean_energy.flatten()[::100], mask_np.flatten()[::100], 
                                      alpha=0.3, s=1)
            axes[2, 2].set_title('Mask vs Clean Energy')
            axes[2, 2].set_xlabel('Clean Spectrogram Value')
            axes[2, 2].set_ylabel('Mask Value')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
        
    return fig


if __name__ == '__main__':
    # 1. Check raw audio differences first
    audio_stats = check_raw_audio_difference(
        data_dir='/scratch/egbueze.m/reverb_dataset',
        num_samples=50
    )
    
    # 2. Check spectrogram differences
    spec_stats = check_spectrogram_difference(
        spec_dir='/scratch/egbueze.m/precomputed_specs_db120',
        num_samples=50
    )
    
    # 3. Load and diagnose model
    model = UNet(in_channels=1, out_channels=1).to('cuda')
    checkpoint = torch.load('/scratch/egbueze.m/checkpoints_db120/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 4. Model diagnosis
    val_dataset = PrecomputedDataset('/scratch/egbueze.m/precomputed_specs_db120')
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    diagnostics = diagnose_model_output(model, val_loader)
    
    # 5. Visualization
    visualize_model_behavior(model, val_loader, save_path='model_analysis.png')
