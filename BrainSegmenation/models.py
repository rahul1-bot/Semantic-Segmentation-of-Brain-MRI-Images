from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


#@: Model 1: UNet   
class UNet(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super(UNet, self).__init__()
        self.conv_down1 = UNet._double_conv(3, 64)
        self.conv_down2 = UNet._double_conv(64, 128)
        self.conv_down3 = UNet._double_conv(128, 256)
        self.conv_down4 = UNet._double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.conv_up3 = UNet._double_conv(256 + 512, 256)
        self.conv_up2 = UNet._double_conv(128 + 256, 128)
        self.conv_up1 = UNet._double_conv(128 + 64, 64)
        
        self.last_conv = nn.Conv2d(64, n_classes, kernel_size=1)
        
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv_down1(x)  
        x = self.maxpool(conv1)     
        conv2 = self.conv_down2(x)  
        x = self.maxpool(conv2)     
        conv3 = self.conv_down3(x)  
        x = self.maxpool(conv3)     
        x = self.conv_down4(x)      
        x = self.upsample(x)        
          
        x = torch.cat([x, conv3], dim=1) 
        
        x = self.conv_up3(x) 
        x = self.upsample(x)  
        x = torch.cat([x, conv2], dim=1) 

        x = self.conv_up2(x)  
        x = self.upsample(x)    
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.conv_up1(x)
        
        out = self.last_conv(x) 
        out = torch.sigmoid(out)
        
        return out
    

    
    @staticmethod
    def _double_conv(in_channels: int, out_channels: int) -> nn.Sequential():
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )




class ConvReluUpsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                                         upsample: Optional[bool] = False) -> None:
        super(ConvReluUpsample, self).__init__()
        self.upsample = upsample
        self.make_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        if self.upsample:
            x = self.make_upsample(x)
        return x




class SegmentationBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                                         n_upsamples: optional[int] = 0) -> None:
        super(SegmentationBlock, self).__init__()
        blocks: list[object] = [
            ConvReluUpsample(in_channels, out_channels, upsample=bool(n_upsamples))
        ]
        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(ConvReluUpsample(out_channels, out_channels, upsample=True))
        self.block = nn.Sequential(*blocks)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)





#@: Model 2: Convolutional Feature Pyramid network 
class FeaturePyramid_Network(nn.Module):
    def __init__(self, n_classes: Optional[int] = 1, pyramid_channels: Optional[int] = 256, 
                                                     segmentation_channels: Optional[int] = 256) -> None:
        super(FeaturePyramid_Network, self).__init__()
        self.conv_down1 = FeaturePyramid_Network._double_conv(3, 64)
        self.conv_down2 = FeaturePyramid_Network._double_conv(64, 128)
        self.conv_down3 = FeaturePyramid_Network._double_conv(128, 256)
        self.conv_down4 = FeaturePyramid_Network._double_conv(256, 512)        
        self.conv_down5 = FeaturePyramid_Network._double_conv(512, 1024)   
        self.maxpool = nn.MaxPool2d(2)
        
        self.toplayer = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.latlayer1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        
        self.seg_blocks = nn.ModuleList([
            SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=n_upsamples)
            for n_upsamples in [0, 1, 2, 3]
        ])
        
        self.last_conv = nn.Conv2d(256, n_classes, kernel_size=1, stride=1, padding=0)
        


    def upsample_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _,_,H,W = y.size()
        upsample = nn.Upsample(size=(H,W), mode='bilinear', align_corners=True) 
        return upsample(x) + y
    


    def upsample(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        sample = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)
        return sample(x)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.maxpool(self.conv_down1(x))
        c2 = self.maxpool(self.conv_down2(c1))
        c3 = self.maxpool(self.conv_down3(c2))
        c4 = self.maxpool(self.conv_down4(c3))
        c5 = self.maxpool(self.conv_down5(c4)) 
        
        p5 = self.toplayer(c5) 
        p4 = self.upsample_add(p5, self.latlayer1(c4)) 
        p3 = self.upsample_add(p4, self.latlayer2(c3))
        p2 = self.upsample_add(p3, self.latlayer3(c2)) 
        
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        
        _, _, h, w = p2.size()
        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p2, p3, p4, p5])]
        
        out = self.upsample(self.last_conv(sum(feature_pyramid)), 4 * h, 4 * w)
        
        out = torch.sigmoid(out)
        return out

    

    @staticmethod
    def _double_conv(in_channels: int, out_channels: int) -> nn.Sequential():
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )




class ConvRelu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                                         kernel: int, 
                                         padding: int) -> None:
        super(ConvRelu, self).__init__()
        self.convrelu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
            nn.ReLU(inplace=True)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor :
        x = self.convrelu(x)
        return x




class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(DecoderBlock, self).__init__()
        self.conv1 = ConvRelu(in_channels, in_channels // 4, 1, 0)
        self.deconv = nn.ConvTranspose2d(
                              in_channels // 4, 
                              in_channels // 4, 
                              kernel_size=4,
                              stride=2, 
                              padding=1, 
                              output_padding=0
                            )
        self.conv2 = ConvRelu(in_channels // 4, out_channels, 1, 0)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.deconv(x)
        x = self.conv2(x)
        return x



#@: Model 3: UNet with ResNext-50
class ResNeXtUNet(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super(ResNeXtUNet, self).__init__()
        self.base_model = resnext50_32x4d(pretrained=True)
        self.base_layers = list(self.base_model.children())
        filters: list[int] = [4*64, 4*128, 4*256, 4*512]
        
        self.encoder0 = nn.Sequential(*self.base_layers[:3])
        self.encoder1 = nn.Sequential(*self.base_layers[4])
        self.encoder2 = nn.Sequential(*self.base_layers[5])
        self.encoder3 = nn.Sequential(*self.base_layers[6])
        self.encoder4 = nn.Sequential(*self.base_layers[7])

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.last_conv0 = ConvRelu(256, 128, 3, 1)
        self.last_conv1 = nn.Conv2d(128, n_classes, 3, padding=1)
                       


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder0(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
    
        out = self.last_conv0(d1)
        out = self.last_conv1(out)
        out = torch.sigmoid(out)    
        return out



#@: Driver code
if __name__.__contains__('__main__'):
    pass