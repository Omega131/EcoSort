import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Attention Module"""
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(1, in_channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.GELU(),
            nn.Linear(reduced_channels, in_channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MobileMQA(nn.Module):
    """Extremely lightweight Mobile Multi-Query Attention"""
    def __init__(self, in_channels, head_dim=8):
        super().__init__()
        self.q = nn.Conv2d(in_channels, head_dim, kernel_size=1, bias=False)
        self.k = nn.Conv2d(in_channels, head_dim, kernel_size=1, bias=False)
        self.v = nn.Conv2d(in_channels, head_dim, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(head_dim, in_channels, kernel_size=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.scale = head_dim ** -0.5
        
    def forward(self, x):
        B, C, H, W = x.size()
        
        q = self.q(x).view(B, -1, H*W).permute(0, 2, 1) # B, H*W, head_dim
        
        # Pool KV to save computation (1x1 spatial resolution for K and V)
        x_pooled = self.pool(x)
        k = self.k(x_pooled).view(B, -1, 1)             # B, head_dim, 1
        v = self.v(x_pooled).view(B, -1, 1).permute(0, 2, 1) # B, 1, head_dim
        
        attn = torch.matmul(q, k) * self.scale          # B, H*W, 1
        attn = F.softmax(attn, dim=1)
        
        out = torch.matmul(attn, v)                     # B, H*W, head_dim
        out = out.permute(0, 2, 1).view(B, -1, H, W)
        out = self.proj(out)
        return x + out

class UniversalInvertedBottleneck(nn.Module):
    """MobileNetV4 style Universal Inverted Bottleneck (UIB)"""
    def __init__(self, in_channels, out_channels, stride, expand_ratio, uib_type='standard', use_mqa=False):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        
        layers = []
        
        # ConvNeXt / ExtraDW start with Depthwise
        if uib_type in ['convnext', 'extra_dw']:
            start_stride = stride if uib_type in ['convnext', 'extra_dw'] else 1
            layers.extend([
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=start_stride, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.GELU()
            ])
            
        # Pointwise Expansion
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU()
            ])
            
        # Middle Depthwise
        if uib_type in ['standard', 'extra_dw']:
            middle_stride = stride if uib_type == 'standard' else 1
            layers.extend([
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=middle_stride, padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU()
            ])
            
        # Attention
        if use_mqa:
            layers.append(MobileMQA(hidden_dim))
        else:
            layers.append(SEBlock(hidden_dim))
            
        # Pointwise Projection
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class EcoNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU()
        )
        
        # MobileNetV4 Block Configs:
        # out_c, stride, expand, uib_type, use_mqa
        block_configs = [
            (16, 1, 1, 'convnext', False),
            (24, 1, 4, 'extra_dw', False), # Late-stage downsampling (delayed stride 2)
            (24, 2, 4, 'extra_dw', False),
            (32, 1, 4, 'standard', False), 
            (32, 2, 4, 'standard', False),
            (48, 1, 5, 'extra_dw', False), 
            (48, 2, 5, 'extra_dw', False),
            (64, 1, 6, 'standard', True),  # Using MobileMQA
            (64, 2, 6, 'standard', True),  # Using MobileMQA
            (96, 1, 5, 'extra_dw', True),  # Using MobileMQA
        ]
        
        layers = []
        in_c = 16
        for out_c, s, exp, uib, mqa in block_configs:
            layers.append(UniversalInvertedBottleneck(
                in_channels=in_c, out_channels=out_c, stride=s, expand_ratio=exp,
                uib_type=uib, use_mqa=mqa
            ))
            in_c = out_c
            
        self.blocks = nn.Sequential(*layers)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(96, 320, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(320),
            nn.GELU()
        )
        
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
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    model = EcoNet(num_classes=6)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    dummy_image = torch.randn(1, 3, 96, 96)
    output = model(dummy_image)
    print(f"Output shape: {output.shape}")
