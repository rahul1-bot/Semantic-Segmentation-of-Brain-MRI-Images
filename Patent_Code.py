from __future__ import annotations
__author__: list[str] = ['Rahul Sawhney']

__doc__: str = r'''
    Patent Title: Semantic Segmentation for Brain MRI images for FLAIR Abnormality Detection
    Patent Abstract: ...
    Patent Published Limk: ...

'''
import warnings, os, copy, time
from tqdm import tqdm
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils



class MRIDataSet(torch.utils.data.Dataset):
    def __init__(self, path: 'dir_path', transform: torchvision.transforms) -> None:
        self.path = path
        self.patients: list[str] = [folder for folder in os.listdir(path) if not folder in ['data.csv', 'README.md']]
        self.transform = transform
        self.masks, self.images = [], []
        
        for patients in self.patients:
            for file in os.listdir(os.path.join(self.path, patients)):
                if 'mask' in file.split('.')[0].split('_'):
                    self.masks.append(os.path.join(self.path, patients, file))
                else:
                    self.images.append(os.path.join(self.path, patients, file))
        
        self.images = sorted(self.images)
        self.masks = sorted(self.masks)
    


    def __len__(self) -> int:
        return len(self.images)
    


    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index]
        mask = self.masks[index]
        if self.transform is not None:
            image = Image.open(image)
            image = self.transform(image)
            mask = Image.open(mask)
            mask = self.transform(mask)
        return image, mask




class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                                         kernel_size: int = 3, 
                                         padding: int = 1) -> None:
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
    def __init__(self, channel1: int, channel2: int, 
                                      kernel_size: int = 3, 
                                      padding: int = 1) -> None:
        super(StackEncoder, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size= 2, stride= 2)
        layers: dict[str, object] = {
            'convBlock_1': ConvBlock(channel1, channel2, kernel_size, padding),
            'convBlock_2': ConvBlock(channel2, channel2, kernel_size, padding)
        }
        self.block = nn.Sequential(*layers.values())
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        big_out = self.block(x)
        poolout = self.pool(big_out)
        return big_out, poolout




class StackDecoder(nn.Module):
    def __init__(self, big_channel: int, channel1: int, 
                                         channel2: int, 
                                         kernel_size: int = 3, 
                                         padding: int = 1) -> None:
        super(StackDecoder, self).__init__()
        layers: dict[str, object] = {
            'convBlock_1': ConvBlock(channel1 + big_channel, channel2, kernel_size, padding),
            'convBlock_2': ConvBlock(channel2, channel2, kernel_size, padding),
            'convBlock_3': ConvBlock(channel2, channel2, kernel_size, padding)
        }
        self.block = nn.Sequential(*layers.values())
    


    def forward(self, x: torch.Tensor, down_tensor: torch.Tensor) -> torch.Tensor:
        _, channels, height, width = down_tensor.size()
        x = F.upsample(x, size= (height, width), mode= 'bilinear')
        x = torch.cat([x, down_tensor], 1)
        x = self.block(x)
        return x
    



class UNet(nn.Module):
    def __init__(self, input_shape: tuple[int, ...]) -> None:
        super(UNet, self).__init__()
        channel, height, width = input_shape

        
        self.stack_encoder_1 = StackEncoder(channel1= channel, channel2= 12, kernel_size= 3)
        self.stack_encoder_2 = StackEncoder(channel1= 12, channel2= 24, kernel_size= 3)
        self.stack_encoder_3 = StackEncoder(channel1= 24, channel2= 46, kernel_size= 3)
        self.stack_encoder_4 = StackEncoder(channel1= 46, channel2= 64, kernel_size= 3)
        self.stack_encoder_5 = StackEncoder(channel1= 64, channel2= 128, kernel_size= 3)
        
        self.centre_layer = ConvBlock(in_channels= 128, out_channels= 128, kernel_size= 3, padding= 1)
        
        self.stack_decoder_5 = StackDecoder(big_channel= 128, channel1= 128, channel2= 64, kernel_size= 3)
        self.stack_decoder_4 = StackDecoder(big_channel= 64, channel1= 64, channel2= 46, kernel_size= 3)
        self.stack_decoder_3 = StackDecoder(big_channel= 46, channel1= 46, channel2= 24, kernel_size= 3)
        self.stack_decoder_2 = StackDecoder(big_channel= 24, channel1= 24, channel2= 12, kernel_size= 3)
        self.stack_decoder_1 = StackDecoder(big_channel= 12, channel1= 12, channel2= 12, kernel_size= 3)
        
        self.last_conv = nn.Conv2d(in_channels= 12, out_channels= 1, kernel_size= 1, bias= True)

        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down1, out = self.stack_encoder_1(x)
        down2, out = self.stack_encoder_2(out)
        down3, out = self.stack_encoder_3(out)
        down4, out = self.stack_encoder_4(out)
        down5, out = self.stack_encoder_5(out)

        out = self.centre_layer(out)

        up5 = self.stack_decoder_5(out, down5)
        up4 = self.stack_decoder_4(up5, down4)
        up3 = self.stack_decoder_3(up4, down3)
        up2 = self.stack_decoder_2(up3, down2)
        up1 = self.stack_decoder_1(up2, down1)

        out = self.last_conv(up1)
        return out




class DiceBCELoss(nn.Module):
    def __init__(self, weight: float = None, size_avg: bool = True) -> None:
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
        




def dice_score(y_true, y_pred, smoothing=1e-6):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum()
    return (2. * intersection + smoothing) / (union + smoothing)



class DiceScore():
    def __init__(self, threshold=0.5, smoothing=1e-6):
        self.name = 'DSC'
        self.smoothing = smoothing
        self.target = 'max'
        self.threshold = 0.5
        

    def __call__(self, y_true, y_pred):
        y_pred[y_pred >= self.threshold] = 1.
        y_pred[y_pred <= self.threshold] = 0.
        
        dscs = np.array(list(map(dice_score, y_true, y_pred, [self.smoothing for _ in range(y_pred.shape[0])])))
        
        return np.mean(dscs)



class Model():
    def __init__(self, net: 'model', criterion: object, 
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
    


    def train_validate(self, history: bool = False) -> dict[str, float]| None:
        since = time.time()
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

                for images, masks in self.dataloaders[phase]:
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
        
        time_elapsed = time.time() - since
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
        train_loss_list = [float(x) for x in history_resnet18['train_loss']]
        test_loss_list =  [float(x) for x in history_resnet18['val_loss']]
        plt.plot(train_loss_list, '-bx')
        plt.plot(train_loss_list, '-bx')
        plt.plot(test_loss_list, '-rx')
        plt.title('Model Loss Plot')
        plt.xlabel('No of Epoch')
        plt.ylabel('Loss')
        plt.legend(['train', 'validation'], loc= 'best')
        plt.show()




# driver code
if __name__.__contains__('__main__'):
    path: 'dir_path' = 'C:\\Users\\RAHUL\\OneDrive\\Desktop\\brain_segmentation\\kaggle_3m'
    
    transform_list: list[object] = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(256),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ]) 

    dataset = MRIDataSet(path, transform_list)
    for image, mask in dataset:
        print(image.shape)
        print(mask.shape)
        break

    train_data, test_data = torch.utils.data.random_split(dataset, [3600, 329])
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size= 4, shuffle= True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size= 4)

    dataloaders: dict[str, object] = {
        'train': train_loader,
        'test': test_loader
    }
    dataset_sizes: dict[str, int] = {
        'train': len(train_data),
        'test': len(test_data)
    }

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = UNet((3, 256, 256)).to(device)
    
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr= 1e-3)
    metric = DiceScore()

    image_segmentor_Model = Model(net= model, criterion= criterion, 
                                              optimizer= optimizer, 
                                              num_epochs= 250, 
                                              dataloaders= dataloaders,
                                              dataset_sizes= dataset_sizes,
                                              device= device,
                                              metric= metric) 

                                              
    
    image_segmentor_Model.train_validate()
    image_segmentor_Model.train_ValAcc()
    image_segmentor_Model.train_valLoss()

 

