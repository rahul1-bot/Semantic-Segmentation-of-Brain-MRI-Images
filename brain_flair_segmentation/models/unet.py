from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Any, Iterable, Union



class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                                         kernel_size: Optional[int] = 3, 
                                         padding: Optional[int] = 1) -> None:
        super(ConvBlock, self).__init__()
        layers: dict[str, torch.nn] = {
            'conv': nn.Conv2d(in_channels, out_channels, kernel_size, padding= padding, bias= False),
            'batch_norm': nn.BatchNorm2d(out_channels, eps= 1e-4),
            'relu': nn.ReLU(inplace= True)
        }
        self.block = nn.Sequential(*layers.values())
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)




class StackEncoder(nn.Module):
    def __init__(self, channel_one: int, channel_two: int, kernel_size: Optional[int] = 3, padding: Optional[int] = 1) -> None:
        super(StackEncoder, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size= 2, stride= 2)
        layers: dict[str, torch.nn] = {
            'conv_one': ConvBlock(channel_one, channel_two, kernel_size, padding),
            'conv_two': ConvBlock(channel_two, channel_two, kernel_size, padding)
        }
        self.block = nn.Sequential(*layers.values())
    
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        big_out: torch.Tensor = self.block(x)
        pool_out: torch.Tensor = self.max_pool(big_out)
        return big_out, pool_out
        
        



class StackDecoder(nn.Module):
    def __init__(self, big_channel: int, channel_one: int, channel_two: int, 
                                                           kernel_size: Optional[int] = 3, 
                                                           padding: Optional[int] = 1) -> None:
        super(StackDecoder, self).__init__()
        layers: dict[str, torch.nn] = {
            'conv_one': ConvBlock(channel_one + big_channel, channel_two, kernel_size, padding),
            'conv_two': ConvBlock(channel_two, channel_two, kernel_size, padding),
            'conv_three': ConvBlock(channel_two, channel_two, kernel_size, padding)
        }
        self.block = nn.Sequential(*layers.values())
    
    
    
    def forward(self, x: torch.Tensor, down_tensor: torch.Tensor) -> torch.Tensor:
        _, channels, height, width = down_tensor.size()
        x = F.upsample(x, size= (height, width), mode= 'bilinear')
        x = torch.cat([x, down_tensor], 1)
        x = self.block(x)
        return x 
    
       
       


class UNet(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int]) -> None:
        super(UNet, self).__init__()
        channel, height, width = input_shape
        
        self.down_one = StackEncoder(channel, 12)
        self.down_two = StackEncoder(12, 24)
        self.down_three = StackEncoder(24, 46)
        self.down_four = StackEncoder(46, 64)
        self.down_five = StackEncoder(64, 128)
        
        self.centre = ConvBlock(128, 128)
        
        self.up_five = StackDecoder(128, 128, 64)
        self.up_four = StackDecoder(64, 64, 46)
        self.up_three = StackDecoder(46, 46, 24)
        self.up_two = StackDecoder(24, 24, 12)
        self.up_one = StackDecoder(12, 12, 12)
        self.conv = nn.Conv2d(12, 1, kernel_size= 1, bias= True)
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_one, out = self.down_one(x)
        down_two, out = self.down_two(out)
        down_three, out = self.down_three(out)
        down_four, out = self.down_four(out)
        down_five, out = self.down_five(out)
        
        out = self.centre(out)
        
        up_five = self.up_five(out, down_five)
        up_four = self.up_four(up_five, down_four)
        up_three = self.up_three(up_four, down_three)
        up_two = self.up_two(up_three, down_two)
        up_one = self.up_one(up_two, down_one)
        
        out = self.conv(up_one)
        return out
    
    