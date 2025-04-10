import torch
from torch import nn


# Define the denoising autoencoder architecture with binary output
class BinaryDenoisingAutoencoder(nn.Module):
    def __init__(self, input_channels=3):
        super(BinaryDenoisingAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Reduce spatial dimensions
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # Further reduce spatial dimensions
        )
        
        # Decoder - outputs a single channel binary image
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # Upsample
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # Upsample
            
            # Output a single channel binary image (1 = white, 0 = black)
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            # Using sigmoid here - we'll threshold during inference for binary output
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def get_binary_output(self, x, threshold=0.5):
        """Get thresholded binary output (0 or 1)"""
        with torch.no_grad():
            continuous_output = self.forward(x)
            binary_output = (continuous_output > threshold).float()
            return binary_output


