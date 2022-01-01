from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Any, Union
from torchvision.models import resnext50_32x4d
import warnings
warnings.simplefilter("ignore")



class ConvRelu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: Union[int, tuple[int, int]], padding: int) -> None:
        super(ConvRelu, self).__init__()
        layers: dict[str, nn.modules] = {
            'conv_1': nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
            'relu': nn.ReLU(inplace= True)
        }
        self.block = nn.Sequential(*layers.values())


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
    
    


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(DecoderBlock, self).__init__()
        layers: dict[str, nn.modules] = {
            'conv_1': ConvRelu(in_channels, in_channels // 4, 1, 0),
            'de_conv': nn.ConvTranspose2d(in_channels= in_channels // 4, 
                                          out_channels= in_channels // 4, 
                                          kernel_size= 4,
                                          stride= 2, 
                                          padding= 1, 
                                          output_padding= 0),
            'conv_2': ConvRelu(in_channels // 4, out_channels, 1, 0)
        }
        self.block = nn.Sequential(*layers.values())
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)




class ResNeXtUNet(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super(ResNeXtUNet, self).__init__()
        self.base_model = resnext50_32x4d(pretrained=True)
        self.base_layers = list(self.base_model.children())
        filters: list[int] = [
            4 * 64, 4 * 128, 4 * 256, 4 * 512
        ]
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
        x: torch.Tensor = self.encoder0(x)
        e1: torch.Tensor = self.encoder1(x)
        e2: torch.Tensor = self.encoder2(e1)
        e3: torch.Tensor = self.encoder3(e2)
        e4: torch.Tensor = self.encoder4(e3)
        
        d4: torch.Tensor = self.decoder4(e4) + e3
        d3: torch.Tensor = self.decoder3(d4) + e2
        d2: torch.Tensor = self.decoder2(d3) + e1
        d1:torch.Tensor = self.decoder1(d2)
        
        out: torch.Tensor = self.last_conv0(d1)
        out: torch.Tensor = self.last_conv1(out)
        out: torch.Tensor = torch.sigmoid(out)
        return out
    
    
    
    
#@: Driver Code
if __name__.__contains__('__main__'):
    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    resnext_unet = ResNeXtUNet(n_classes= 1).to(device)
    result: torch.Tensor = resnext_unet(torch.rand(1, 3, 256, 256).to(device))
    print(result.shape)