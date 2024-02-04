import torch
import torch.nn as nn

from collections import OrderedDict

class DownsampleInput(nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 r = 2,
                ):
        super().__init__()
        inter_channels = int(in_channels*r)
        
        self.pwconv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(inter_channels, eps=1e-6)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(inter_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.pwconv1(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = self.act(x)
        x = self.pwconv2(x)

        return x

## https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/utils.py
class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2,3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
    
# Features Agregators
class LocalAggregator(nn.Module):
    
    def __init__(self,
                 channels,
                 r = 2,
                 upsample = False,
                 drop_path=0.,
                ):
        super().__init__()
        inter_channels = int(channels*r)
        self.use_upsample = upsample

        self.alpha = nn.Parameter(torch.zeros(1))
    
        if self.use_upsample:
            self.upsample = nn.ConvTranspose2d(channels, channels, kernel_size=(2,2), stride=2, groups=channels)
            self.normup = nn.LayerNorm(channels, eps=1e-6)
        
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.pwconv1 = nn.Conv2d(channels, inter_channels, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(inter_channels, channels, kernel_size=1)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.sigmoid  = nn.Sigmoid()
        
    def forward(self, x):
        
        if self.use_upsample:
            x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            x = self.normup(x)
            x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
            x = self.upsample(x)
        
        x_hat = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x_hat = self.norm(x_hat)
        x_hat = x_hat.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x_hat = self.pwconv1(x_hat)
        x_hat = self.act(x_hat)
        x_hat = self.pwconv2(x_hat)

        s =  self.sigmoid(x_hat)

        return (1-self.alpha)*self.drop_path(torch.mul(x, s)) + self.alpha*x

class GlobalAggregator(nn.Module):
    
    def __init__(self,
                 channels,
                 r = 4,
                 downsample = False,
                 drop_path=0.,
                ):
        super().__init__()
        inter_channels = int(channels*r)
        self.use_downsample = downsample

        self.alpha = nn.Parameter(torch.zeros(1))
        
        if self.use_downsample:
            self.downsample = nn.Conv2d(channels, channels, 3, stride=2, padding=1, groups=channels, bias=False)
            self.normup = nn.LayerNorm(channels, eps=1e-6)
        
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=3,  stride=1, padding=1, groups=channels, bias=False)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        
        self.pwconv1 = nn.Conv2d(channels, inter_channels, kernel_size=1)
        self.act = nn.GELU()
        self.gc = GRN(inter_channels)
        self.pwconv2 = nn.Conv2d(inter_channels, channels, kernel_size=1)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.sigmoid  = nn.Sigmoid()
        
    def forward(self, x):
        
        if self.use_downsample:
            x = self.downsample(x)
            x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            x = self.normup(x)
            x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        
        x_hat = self.dwconv(x)
        x_hat = x_hat.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x_hat = self.norm(x_hat)
        x_hat = x_hat.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x_hat = self.pwconv1(x_hat)
        x_hat = self.act(x_hat)
        x_hat = self.gc(x_hat)
        x_hat = self.pwconv2(x_hat)

        s =  self.sigmoid(x_hat)

        return (1-self.alpha)*self.drop_path(torch.mul(x, s)) + self.alpha*x
    
# Local-Global Feature Fusion Neck (LGFFN)
class LGFFN(nn.Module):
    
    def __init__(self, 
                 channels: int,
                 conv_channels: list[int],
                 first_time: bool = False):
        
        super().__init__()

        # Convs blocks
        ## Inner Nodes
        ### P31
        self.p31_gca_p30 = GlobalAggregator(channels)
        self.p31_lca_p40 = LocalAggregator(channels, upsample = True)
        ### P21
        self.p21_gca_p20 = GlobalAggregator(channels)
        self.p21_lca_p31 = LocalAggregator(channels, upsample = True)
        
        ## Outer Nodes
        ### P12
        self.p12_gca_p10 = GlobalAggregator(channels)
        self.p12_lca_p21 = LocalAggregator(channels, upsample = True)
        
        ### P22
        self.p22_lca_p20 = LocalAggregator(channels)
        self.p22_lca_P21 = LocalAggregator(channels)
        self.p22_gca_p12 = GlobalAggregator(channels, downsample=True)
        
        ### P32
        self.p32_lca_p30 = LocalAggregator(channels)
        self.p32_lca_P31 = LocalAggregator(channels)
        self.p32_gca_p22 = GlobalAggregator(channels, downsample=True)
        
        ### P42
        self.p42_lca_p40 = LocalAggregator(channels)
        self.p42_gca_p32 = GlobalAggregator(channels, downsample=True)

        self.first_time = first_time
        if self.first_time:
            self.p4_down_channel = DownsampleInput(in_channels=conv_channels[3] , out_channels=channels)
            self.p3_down_channel = DownsampleInput(in_channels=conv_channels[2] , out_channels=channels)
            self.p2_down_channel = DownsampleInput(in_channels=conv_channels[1] , out_channels=channels)
            self.p1_down_channel = DownsampleInput(in_channels=conv_channels[0] , out_channels=channels)

    def forward(self, inputs):
        """
        illustration of a minimal CABiFPN unit
            P4_0 --------------------------> P4_2 -------->
               |-------------|                ↑
                             ↓                |
            P3_0 ---------> P3_1 ----------> P3_2 -------->
               |-------------|---------------↑ ↑
                             ↓                 |
            P2_0 ---------> P2_1 ----------> P2_2 -------->
               |-------------|---------------↑ ↑
                             |---------------↓ |
            P1_0 --------------------------> P1_2 -------->
        """
        
        if isinstance(inputs, OrderedDict): inputs = list(inputs.values())
        
        if self.first_time:
            p1, p2, p3, p4 = inputs

            p4_0 = self.p4_down_channel(p4)
            p3_0 = self.p3_down_channel(p3)
            p2_0 = self.p2_down_channel(p2)
            p1_0 = self.p1_down_channel(p1)

        else:
            p1_0, p2_0, p3_0, p4_0 = inputs


        ## Nodes
        ### Inner Nodes
        p3_1 = self.p31_gca_p30(p3_0) + self.p31_lca_p40(p4_0)
        p2_1 = self.p21_gca_p20(p2_0) + self.p21_lca_p31(p3_1)
        
        ### Outer Nodes
        p1_2 = self.p12_gca_p10(p1_0) + self.p12_lca_p21(p2_1)
        p2_2 = self.p22_lca_p20(p2_0) + self.p22_lca_P21(p2_1) + self.p22_gca_p12(p1_2)
        p3_2 = self.p32_lca_p30(p3_0) + self.p32_lca_P31(p3_1) + self.p32_gca_p22(p2_2)
        p4_2 = self.p42_lca_p40(p4_0) + self.p42_gca_p32(p3_2)
        
        features = [p1_2, p2_2, p3_2, p4_2]

        return OrderedDict(zip([f'P{i}' for i in range(len(features))], features))