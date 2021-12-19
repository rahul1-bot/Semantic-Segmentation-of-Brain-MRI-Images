from __future__ import annotations

__authors__: list[str] = ['Rahul Sawhney', 'Aabha Malik', 'Shubham Garg', 'Harsh Sharma']

__authors_email__: dict[str, str] = {
    'Rahul Sawhney': 'sawhney.rahulofficial@outlook.com',
    'Aabha Malik': 'aabhamalik30@gmail.com',
    'Shubham Garg': '',
    'Harsh Sharma': ''
}

__authors_qualifications__: dict[str, str] = {
    x: 'Btech CSE, Amity University, Noida' 
    for x in ['Rahul Sawhney', 'Aabha Malik', 'Shubham Garg', 'Harsh Sharma']    

}


__license__: str = r'''
    MIT License
    Copyright (c) 2021 Rahul Sawhney
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
'''

__doc__: str = r'''
    >>> Paper Title: 
            **Semantic Segmentation of brain Images from UNet, feature Pyramid Network 
    
    >>> Paper Abstract: 
            Brain Tumor segmentation is an essential method in medical image processing. 
            Early medical diagnosis of brain tumors plays an essential function in enhancing 
            treatment possibilities and increases the survival rate of the patients. The 
            most challenging and time taking work is the manual segmentation of the brain 
            tumors for cancer diagnosis from large quantity of MRI images produced in 
            scientific routine. Recently, automated segmentation utilizing deep learning 
            techniques showed popular considering that these techniques accomplish cutting
            edge outcomes and resolve this issue much better than other approaches. 
            Deep Learning techniques can likewise make it possible for effective processing
            and unbiased assessment of the big quantities of MRI-based image information. 
            There are variety of existing evaluation papers, concentrating on conventional 
            techniques for MRI-based brain tumor image segmentation. In this paper, we have 
            used UNet model along with this the **AdamW and **AdaMax are used as optimizer. 
            Criterion used is IOU (Intersection Over Union) and the metric utilized for evaluation
            is Dice Score. 
    
    >>> Paper Keywords: 
        Lower-grade gliomas detection, Brain Semantic Segmentation, Image Recogonition, UNet, Feature-Detector-Network
'''



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os, time, copy
from tqdm import tqdm
from typing import Optional, Union, ClassVar
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, utils
from PIL import Image
import segmentation_models_pytorch as smp
from torchvision.models import resnext50_32x4d
from torch.optim.lr_scheduler import ReduceLROnPlateau




class BrainSegmentorData(torch.utils.data.Dataset):
    '''Custom Brain Data-Object'''
    def __init__(self, path: str, transform: torchvision.transforms) -> None:
        self.path = path
        self.transform = transform
        self.patients: list[str] = [
            folder for folder in os.listdir(self.path) if folder not in ['data.csv', 'README.md']
        ]
        self.csv_data: pd.DataFrame = pd.read_csv(os.path.join(self.path, 'data.csv'))
        self.masks: list[str] = []
        self.images: list[str] = []
        
        for patient in self.patients:
            for folder in os.listdir(os.path.join(self.path, patient)):
                if 'mask' in folder.split('.')[0].split('_'):
                    self.masks.append(os.path.join(self.path, patient, folder))
                else:
                    self.images.append(os.path.join(self.path, patient, folder))
        
        self.images.sort()
        self.masks.sort()
        self.brain_dict: dict[str, list[str]] = {'Patient': self.images, 'Images': self.images, 'Mask': self.masks}
        self.brain_dataframe: pd.DataFrame = pd.DataFrame.from_dict(self.brain_dict)
        self.brain_dataframe['diagnosis'] = self.brain_dataframe['Mask'].apply(lambda mask_path: self._diagnosis(mask_path))
        
    
    
    def __repr__(self) -> str(dict[str, str]):
        return str({
            x: y for x, y in zip(['Module', 'Name', 'ObjectID'], [self.__module__, type(self).__name__, hex(id(self))])
        })
        
    
    
    def __str__(self) -> str(dict[str, str]):
        return str({
            x: y for x, y in zip(['Path', 'Transform'], [self.path, self.transform])
        })
    
        
    
    def __len__(self) -> int:
        return len(self.images)
    
    
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.images[index], self.masks[index]
        if self.transform is not None:
            image, mask = Image.open(image_path), Image.open(mask_path)
            image, mask = self.transform(image), self.transform(mask)
        return image, mask
    
    
    
    def _diagnosis(self, mask_path: str) -> int:
        value: int = np.max(Image.open(mask_path))
        return 1 if value > 0 else 0 



    def get_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.csv_data, self.brain_dataframe
     




class BrainAnalysis:
    '''Brain MetaData Analysis'''
    def __init__(self, brain_metaData: pd.DataFrame, brain_df: pd.DataFrame) -> None:
        self.brain_metaData = brain_metaData
        self.brain_df = brain_df
        
    
    def __repr__(self) -> str:
        return str({
            x: y for x, y in zip(['Module', 'Name', 'Object_ID'], [self.__module__, type(self).__name__, hex(id(self))])
        })
        
    
    def __str__(self) -> str(dict[str, list[str]]):
        dataKey_map: dict[str, list[str]] = {
            x: y for x, y in zip(['brain_metaData', 'brain_df'], [self.brain_metaData.keys(), self.brain_df.keys()])
        }
        return str(dataKey_map)
        
    
    
    def visualize_images(self, n_images: Optional[int] = 4) -> 'plot':
        ...
    
    
    def diagnosis_plot(self, fig_size: Optional[tuple[int, int]] = (10, 6)) -> 'plot':
        ax = self.brain_df['diagnosis'].value_counts().plot(
            kind= 'bar',
            stacked= True,
            figsize= fig_size,
            color=["violet", "lightseagreen"]
        )
        ax.set_xticklabels(["Positive", "Negative"], rotation= 45, fontsize= 12)
        ax.set_ylabel('Total Images', fontsize = 12)
        ax.set_title("Distribution of data grouped by diagnosis",fontsize = 18, y=1.05)
        plt.show()





class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                                         kernel_size: Optional[int] = 3, 
                                         padding: Optional[int] = 1) -> None:
        super(ConvBlock, self).__init__()
        layers: dict[str, object] = {
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
        layers: dict[str, object] = {
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
        layers: dict[str, object] = {
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
                            



class Feature_Pyramid_Network(nn.Module):
    def __init__(self) -> None:
        super(Feature_Pyramid_Network, self).__init__()
        
    
    def forward(self) -> nn.Module():
        model = smp.FPN(
            encoder_name= 'efficientnet-b7',
            encoder_weights= 'imagenet',
            in_channels= 3,
            classes= 1,
            activation= 'sigmoid',
        )
        return model
    
    
    

class ConvRelu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: Union[int, tuple[int, int]], padding: int) -> None:
        super(ConvRelu, self).__init__()
        layers: dict[str, object] = {
            'conv_1': nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
            'relu': nn.ReLU(inplace= True)
        }
        self.block = nn.Sequential(*layers.values())


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
    
    

class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(DecoderBlock, self).__init__()
        layers: dict[str, object] = {
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
        nn.ConvTranspose2d
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)




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
    
        
        

class DiceBCELoss(nn.Module):
    def __init__(self, weight: Optional[Union[float, None]] = None, size_avg: Optional[bool] = True) -> None:
        super(DiceBCELoss, self).__init__()
    

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth: int = 1) -> torch.Tensor:
        inputs = F.sigmoid(inputs)
        bce_weights: float = 0.5
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth)/ (inputs.sum() + targets.sum() + smooth)
        bce = F.binary_cross_entropy(inputs, targets, reduction= 'mean')
        loss_final = bce * bce_weights + dice_loss * (1 - bce_weights)
        return loss_final  
        



class DiceScore():
    def __init__(self, threshold=0.5, smoothing=1e-6):
        self.name = 'DSC'
        self.smoothing = smoothing
        self.target = 'max'
        self.threshold = 0.5
    
    
    def _dice_score(self, y_true: torch.Tensor, y_pred: torch.Tensor, smoothing: Optional[float]=1e-6) -> torch.Tensor:
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum()
        return (2. * intersection + smoothing) / (union + smoothing)


    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> np.ndarray():
        y_pred[y_pred >= self.threshold] = 1.
        y_pred[y_pred <= self.threshold] = 0.
        
        dscs = np.array(list(map(self._dice_score, y_true, y_pred, [self.smoothing for _ in range(y_pred.shape[0])])))
        
        return np.mean(dscs)



        
    
class EarlyStop:
    def __init__(self, patience: Optional[int] = 6, min_delta: Optional[float] = 0, 
                                                    weights_path: Optional[str] = 'model_weights.pt') -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter: int = 0
        self.best_loss: float = float('inf')
        self.weights_path = weights_path
    
    
    def __repr__(self) -> str(dict[str, str]):
        return str({
            x: y for x, y in zip(['Module', 'Name', 'ObjectID'], [self.__module__, type(self).__name__, hex(id(self))])
        })
        
    
    def __str__(self) -> str(dict[str, str]):
        return str({
            'Patience': self.patience,
            'Min_Delta': self.min_delta,
            'Counter': self.counter,
            'Optimal_loss': self.best_loss,
            'Model_weights': self.weights_path
        })
        
        
    
    def __call__(self, test_loss: float, model: torch.nn.Module) -> bool:
        if self.best_loss - test_loss > self.min_delta:
            self.best_loss = test_loss
            torch.save(model.state_dict(), self.weights_path)
            self.counter = 0
        
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        
        return False
    
    
    
    def load_weights(self, model: torch.nn.Module) -> torch.nn.Module():
        return model.load_state_dict(torch.load(self.weights_path))
    
    



class Model():
    def __init__(self, net: torch.nn.Module, criterion: object, 
                                             optimizer: object, 
                                             num_epochs: int, 
                                             dataloaders: dict[str, object],
                                             dataset_sizes: dict[str, int], 
                                             device: torch.device, 
                                             metric: object = None) -> None:
        super(Model, self).__init__()
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes
        self.device = device
        self.metric = metric
    
    
    
    def __repr__(self) -> str(dict[str, str]):
        return str({
            x: y for x, y in zip(['Module', 'Name', 'ObjectID'], [self.__module__, type(self).__name__, hex(id(self))])
        })
        
    
    
    def __str__(self) -> str(dict[str, str]):
        return str({
            'Net': self.net,
            'Criterion': self.criterion,
            'Optimizer': self.optimizer,
            'Epochs': self.num_epochs,
            'DataLoader_dict': self.dataloaders,
            'Dataset_map': self.dataset_sizes,
            'Device': self.device,
            'Metric': self.metric
            
        })    


    def train_validate(self, history: bool = False) -> Union[None, dict[str, float]]:
        since: float = time.time()
        best_model_wts = copy.deepcopy(self.net.state_dict())
        best_acc: float = 0.0
        self.history: dict[str, list] = {
            x: [] for x in ['train_loss', 'val_loss', 'train_acc', 'val_acc']
        }

        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch + 1}/{self.num_epochs}')
            print('-' * 10)

            for phase in ['train', 'test']:
                if phase == 'train':
                    self.net.train()
                else:
                    self.net.eval()
                
                running_loss: float = 0.0
                running_diceScore: float = 0.0

                for images, masks in tqdm(self.dataloaders[phase]):
                    images = images.to(self.device, dtype= torch.float)
                    masks = masks.to(self.device, dtype= torch.float)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        pred_masks = self.net(images)
                        loss = self.criterion(pred_masks, masks)
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                    
                    running_loss += loss.item() * images.size(0)
                    running_diceScore += self.metric(masks.detach().cpu().numpy(), pred_masks.detach().cpu().numpy()) * images.size(0)

                epoch_loss: float = running_loss/ self.dataset_sizes[phase]
                epoch_acc: float =  running_diceScore/ self.dataset_sizes[phase]
                if phase == 'train':
                    self.history['train_loss'].append(epoch_loss)
                    self.history['train_acc'].append(epoch_acc)
                else:
                    self.history['val_loss'].append(epoch_loss)
                    self.history['val_acc'].append(epoch_acc)
                
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                if phase ==  'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.net.state_dict())

            print()
        
        time_elapsed: float = time.time() - since
        print(f'Training Completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s') 
        print(f'Best Val Acc: {best_acc:.4f}')
        
        if history:
            return self.history


    

    def train_ValAcc(self) -> 'plot':
        train_acc_list = [float(x.cpu().numpy()) for x in self.history['train_acc']]
        test_acc_list = [float(x.cpu().numpy()) for x in self.history['val_acc']]
        plt.plot(train_acc_list, '-bx')
        plt.plot(test_acc_list, '-rx')
        plt.title('Model Accuracy Plot')
        plt.xlabel('No of Epochs')
        plt.ylabel('Accuracy')
        plt.legend(['train', 'validation'], loc= 'best')
        plt.show()



    def train_valLoss(self) -> 'plot':
        train_loss_list = [float(x) for x in self.history['train_loss']]
        test_loss_list =  [float(x) for x in self.history['val_loss']]
        plt.plot(train_loss_list, '-bx')
        plt.plot(train_loss_list, '-bx')
        plt.plot(test_loss_list, '-rx')
        plt.title('Model Loss Plot')
        plt.xlabel('No of Epoch')
        plt.ylabel('Loss')
        plt.legend(['train', 'validation'], loc= 'best')
        plt.show()
    
    
        
        
        

#@: Driver code
if __name__.__contains__('__main__'):
    path: str = 'C:\\Users\\RAHUL\\OneDrive\\Desktop\\brain_segmentation\\kaggle_3m'
    
    transform_list: list[object] = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(256),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ]) 
    
    brain_data: object = BrainSegmentorData(path, transform_list)
    brain_metadata, brain_df = brain_data.get_data()
    
    for image, mask in brain_data:
        print(image.shape)
        print(mask.shape)
        break
    
    train_data, test_data = torch.utils.data.random_split(brain_data, [3600, 329])
    
    train_loader = torch.utils.data.DataLoader(
        dataset= train_data, 
        batch_size= 4, 
        shuffle= True
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset= test_data, 
        batch_size= 4
    )
    
    dataloaders: dict[str, object] = {
        'train': train_loader,
        'test': test_loader
    }
    dataset_sizes: dict[str, int] = {
        'train': len(train_data),
        'test': len(test_data)
    }
    
    brain_analysis = BrainAnalysis(brain_df, train_data)
    brain_analysis.diagnosis_plot()
    device: torch.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    unet = UNet((3, 256, 256)).to(device)
    resnext_unet = ResNeXtUNet(1).to(device)
    fpn_body = Feature_Pyramid_Network()
    fpn = fpn_body().to(device)
    
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(unet.parameters(), lr= 1e-3)
    metric = DiceScore()
    
    model_unet = Model(
        net= unet, 
        criterion= criterion, 
        optimizer= optimizer, 
        num_epochs= 12, 
        dataloaders= dataloaders,
        dataset_sizes= dataset_sizes,
        device= device,
        metric= metric
    ) 
    
    model_resnextUnet = Model(
        net= resnext_unet, 
        criterion= criterion, 
        optimizer= optimizer, 
        num_epochs= 12, 
        dataloaders= dataloaders,
        dataset_sizes= dataset_sizes,
        device= device,
        metric= metric
    )
    
    model_fpn = Model(
        net= fpn, 
        criterion= criterion, 
        optimizer= optimizer, 
        num_epochs= 12, 
        dataloaders= dataloaders,
        dataset_sizes= dataset_sizes,
        device= device,
        metric= metric
    )
    # model_fpn.train_validate()
    # model_unet.train_validate()
    # model_resnextUnet.train_validate()
    
    
    
    
    
    
    