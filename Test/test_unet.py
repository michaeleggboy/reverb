import unittest
import torch

from unet import UNet


class TestUNetArchitecture(unittest.TestCase):
    """Test U-Net model architecture"""
    
    def setUp(self):
        """Create model instance"""
        self.model = UNet(in_channels=1, out_channels=1)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def test_model_creation(self):
        """Test model can be created"""
        self.assertIsInstance(self.model, UNet)
    
    def test_forward_pass_shape(self):
        """Test forward pass produces correct output shape"""
        # Input: batch of 4 spectrograms, 256x256
        x = torch.randn(4, 1, 256, 256)
        
        output = self.model(x)
        
        # Output should match input shape
        self.assertEqual(output.shape, torch.Size([4, 1, 256, 256]))
    
    def test_forward_pass_single_sample(self):
        """Test with single sample (batch size 1)"""
        x = torch.randn(1, 1, 256, 256)
        output = self.model(x)
        
        self.assertEqual(output.shape, torch.Size([1, 1, 256, 256]))
    
    def test_output_no_nan(self):
        """Test that forward pass doesn't produce NaN"""
        x = torch.randn(2, 1, 256, 256)
        output = self.model(x)
        
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_gradient_flow(self):
        """Test that gradients flow through the network"""
        x = torch.randn(2, 1, 256, 256, requires_grad=True)
        
        output = self.model(x)
        loss = output.sum()
        loss.backward()
        
        # Check that input has gradients
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.any(x.grad != 0))
    
    def test_model_on_gpu(self):
        """Test model can be moved to GPU"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        model = UNet(in_channels=1, out_channels=1).to('cuda')
        x = torch.randn(2, 1, 256, 256).to('cuda')
        
        output = model(x)
        
        # Output should be on GPU
        self.assertEqual(output.device.type, 'cuda')
    
    def test_different_batch_sizes(self):
        """Test with various batch sizes"""
        for batch_size in [1, 4, 8, 16]:
            x = torch.randn(batch_size, 1, 256, 256)
            output = self.model(x)
            
            self.assertEqual(output.shape[0], batch_size)
    
    def test_model_parameters_count(self):
        """Test model has reasonable number of parameters"""
        param_count = sum(p.numel() for p in self.model.parameters())
        
        # U-Net with [64, 128, 256, 512] should have ~30-40M parameters
        self.assertGreater(param_count, 1_000_000)  # At least 1M
        self.assertLess(param_count, 100_000_000)   # Less than 100M


class TestUNetTraining(unittest.TestCase):
    """Test U-Net in training context"""
    
    def test_train_mode(self):
        """Test switching between train and eval modes"""
        model = UNet()
        
        # Default is train mode
        self.assertTrue(model.training)
        
        # Switch to eval
        model.eval()
        self.assertFalse(model.training)
        
        # Switch back to train
        model.train()
        self.assertTrue(model.training)
    
    def test_backward_pass(self):
        """Test backward pass works"""
        model = UNet()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Create dummy batch
        reverb = torch.randn(4, 1, 256, 256)
        clean = torch.randn(4, 1, 256, 256)
        
        # Forward pass
        pred = model(reverb)
        loss = criterion(pred, clean)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check that some parameters were updated
        # (gradients should be non-zero)
        has_gradients = any(
            p.grad is not None and torch.any(p.grad != 0)
            for p in model.parameters()
        )
        self.assertTrue(has_gradients)


if __name__ == '__main__':
    unittest.main(verbosity=2)
