import dataset
import unet
import torch
from torch import nn
from torch.utils.data import DataLoader

dataset = dataset.DereverbDataset()
dataloader = DataLoader(
    dataset, 
    batch_size=32, 
    shuffle=True,
    num_workers=4
)

model = unet.UNet(in_channels=1, out_channels=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train(num_epochs):
    for epoch in range(num_epochs):
        for reverb_spec, clean_spec in dataloader:
            pred_spec = model(reverb_spec)
            loss = criterion(pred_spec, clean_spec)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()