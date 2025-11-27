from precomputed_dataset import PrecomputedDataset
from unet import UNet
import torch
import numpy as np
from torch.utils.data import DataLoader


def diagnose_model_output(model, test_loader, device='cuda'):
    """Comprehensive diagnosis of what model is actually doing"""
    
    model.eval()
    diagnostics = {
        'input_ranges': [],
        'output_ranges': [],
        'changes': [],
        'freq_responses': []
    }
    
    with torch.no_grad():
        for reverb, clean in test_loader:
            reverb = reverb.to(device)
            pred = model(reverb)
            
            # 1. Dynamic range analysis
            in_range = (reverb.min().item(), reverb.max().item())
            out_range = (pred.min().item(), pred.max().item())
            diagnostics['input_ranges'].append(in_range)
            diagnostics['output_ranges'].append(out_range)
            
            # 2. How much is model changing?
            change = torch.abs(pred - reverb).mean().item()
            diagnostics['changes'].append(change)
            
            # 3. Frequency response
            # Low frequencies (bottom quarter)
            low_in = reverb[:, :, :64, :].mean().item()
            low_out = pred[:, :, :64, :].mean().item()
            
            # High frequencies (top quarter)
            high_in = reverb[:, :, 192:, :].mean().item()
            high_out = pred[:, :, 192:, :].mean().item()
            
            diagnostics['freq_responses'].append({
                'low_ratio': low_out / (low_in + 1e-6),
                'high_ratio': high_out / (high_in + 1e-6)
            })
            
            break  # Just one batch for quick diagnosis
    
    # Print diagnosis
    print("=== MODEL BEHAVIOR DIAGNOSIS ===")
    print(f"Input range: {np.mean([r[0] for r in diagnostics['input_ranges']]):.3f} to {np.mean([r[1] for r in diagnostics['input_ranges']]):.3f}")
    print(f"Output range: {np.mean([r[0] for r in diagnostics['output_ranges']]):.3f} to {np.mean([r[1] for r in diagnostics['output_ranges']]):.3f}")
    print(f"Average change: {np.mean(diagnostics['changes']):.4f}")
    
    freq = diagnostics['freq_responses'][0]
    print(f"Low freq preservation: {freq['low_ratio']:.2%}")
    print(f"High freq preservation: {freq['high_ratio']:.2%}")
    
    # Identify problems
    if np.mean(diagnostics['changes']) < 0.01:
        print("⚠️ PROBLEM: Model barely changing input (might be copying)")
    if freq['high_ratio'] < 0.5:
        print("⚠️ PROBLEM: Losing too much high frequency (muffled sound)")
    if freq['low_ratio'] < 0.7:
        print("⚠️ PROBLEM: Losing low frequencies (thin sound)")
        
    return diagnostics


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
    diagnostics = diagnose_model_output(model, val_loader, device='cuda')
