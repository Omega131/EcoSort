import torch
import torch.nn as nn
import torch.nn.functional as F
class SEBlock(nn.Module):
    """Squeeze-and-Excitation Attention Module"""
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        # 'Squeeze' phase: Average pool the feature map down to a single pixel per channel (1x1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 'Excitation' phase: Two small linear layers to learn which channels matter
        reduced_channels = max(1, in_channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.GELU(),
            nn.Linear(reduced_channels, in_channels, bias=False),
            nn.Sigmoid() # Squashes output between 0 and 1
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # Scale the original input by the learned attention weights
        return x * y.expand_as(x)
class InvertedResidual(nn.Module):
    """MobileNetV3 style Inverted Residual Block with SE"""
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(in_channels * expand_ratio))
        
        # Skip connection is only possible if input and output sizes match exactly
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        layers = []
        # 1. Pointwise Expansion (only if expand_ratio > 1)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU()
            ])
            
        # 2. Depthwise Convolution
        # Notice `groups=hidden_dim`: This tells PyTorch to apply a separate filter to EACH channel.
        # This is where we save 90% of our parameters compared to a standard CNN!
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        ])
        
        # 3. Squeeze-and-Excitation Attention
        layers.append(SEBlock(hidden_dim))
        
        # 4. Pointwise Linear Projection back to out_channels 
        # (Notice there is NO activation function here, to prevent information loss)
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)  # Skip connection
        else:
            return self.conv(x)

class EcoNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        
        # 1. Initial Convolution (Transforms 3 RGB channels into 16, shrinks 96x96 -> 48x48)
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU()
        )
        
        # 2. Main Inverted Residual Blocks (Increased Configuration)
        block_configs = [
            # (out_channels, stride, expand_ratio)
            (16, 1, 1),
            (24, 2, 4),  
            (24, 1, 4),
            (32, 2, 4),
            (32, 1, 4),
            (48, 2, 4),
            (48, 1, 4),
            (64, 1, 4),
            (64, 1, 4),
            (96, 2, 4),
        ]
        
        layers = []
        in_c = 16
        for out_c, s, exp in block_configs:
            layers.append(InvertedResidual(in_channels=in_c, out_channels=out_c, stride=s, expand_ratio=exp))
            in_c = out_c
            
        self.blocks = nn.Sequential(*layers)
        
        # 3. Final Pointwise Convolution 
        self.final_conv = nn.Sequential(
            nn.Conv2d(96, 320, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(320),
            nn.GELU()
        )
        
        # 4. Classifier Head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4), 
            nn.Linear(320, num_classes)
        )

    def forward(self, x):
        x = self.first_conv(x)
        x = self.blocks(x)
        x = self.final_conv(x)
        x = self.pool(x)
        x = torch.flatten(x, 1) # Flattens from [Batch, Channels, 1, 1] to [Batch, Channels]
        x = self.classifier(x)
        return x

# --- TEST BLOCK ---
# This allows us to run `python model.py` directly to check our architecture!
if __name__ == '__main__':
    model = EcoNet(num_classes=6)
    
    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    
    # Test a dummy image passing through the network
    dummy_image = torch.randn(1, 3, 96, 96)
    output = model(dummy_image)
    print(f"Output shape (Batch Size, Classes): {output.shape}")
