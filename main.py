from __future__ import annotations

__Patent_license__: str = r'''
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

__authors__: list[str] = ['Rahul Sawhney', 'Shubham Garg', 'Harsh Sharma', 'Aabha Malik']

__authors_qualifications__: str = r'''
    Rahul Sawhney: Btech CSE Dept. Amity University, Noida 
    Shubham Garg: Btech CSE Dept. Amity University, Noida
    Harsh Sharma: Btech CSE Dept. Amity University, Noida
    Aabha Malik: Btech CSE Dept. Amity University, Noida
'''

__Patent_doc__: str = r'''
    >>> Patent Topic: Semantic Segmentation of Brain MRI Images for FLAIR Abnormailty Detection  
    >>> Patent Abstract: //...//
    >>> Patent Conclusion: //...//
'''

__loss_doc__: str = r'''
    LOSS = BCE - Log(DICE)
    where,
        1) BCE = Binary Cross Entropy Loss Function
        2) DICE = Custom Dice Coeffient Loss 
'''

__hardware_info__: str = r'''
    CPU: AMD Ryzen 5 3600
    GPU: Nvidia GeForce RTX 2060
    RAM: 32 GB 3600 Mhz
    ROM: Samsung 970 EVO Plus NVME 1 TB
'''

from BrainSegmenation.models import *
from BrainSegmenation.data import *
from BrainSegmenation.utils import *


#@: Driver code
if __name__.__contains__('__main__'):
    # path: 'dir_path' = 'C:\\Users\\RAHUL\\OneDrive\\Desktop\\brain_segmentation\\kaggle_3m'
    
    # transform_list: list[object] = transforms.Compose([
    #     transforms.Resize(128),
    #     transforms.RandomCrop(128),
    #     transforms.RandomVerticalFlip(),
    #     transforms.ToTensor()
    # ]) 
    # dataset = BrainDataset(path, transform_list)
    # for image, mask in dataset:
    #     print(image.shape)
    #     print(mask.shape)
    #     break
    
    # train_data, test_data = torch.utils.data.random_split(dataset, [3600, 329])
    
    # train_loader = torch.utils.data.DataLoader(train_data, batch_size= 1, shuffle= True)
    # test_loader = torch.utils.data.DataLoader(test_data, batch_size= 1)

    # dataloaders: dict[str, object] = {
    #     'train': train_loader,
    #     'test': test_loader
    # }
    # dataset_sizes: dict[str, int] = {
    #     'train': len(train_data),
    #     'test': len(test_data)
    # }

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # # models 
    # unet = UNet(n_classes= 1).to(device)
    # fpn = FeaturePyramid_Network().to(device)
    # rx50 = ResNeXtUNet(n_classes= 1).to(device)
    
    # # Models Optimizers
    # unet_optimizer = torch.optim.Adam(unet.parameters(), lr=1e-3)
    # #fpn_optimizer = torch.optim.Adamax(fpn.parameters(), lr=1e-3)
    # #rx50_optimizer = torch.optim.Adam(rx50.parameters(), lr=5e-4)
  
    # # metrics and loss
    # metric = DiceScore()
    # loss = DiceBCELoss()


    # unet_model = Model(
    #     model= unet,
    #     criterion= loss,
    #     optimizer= unet_optimizer,
    #     metric= metric,
    #     num_epochs= 2,
    #     dataloaders= dataloaders,
    #     dataset_sizes= dataset_sizes,
    #     device= device
    # )

    # # Image Segmentor Models
    # unet_model = Model(
    #     model_name= 'UNet',
    #     model= unet,
    #     train_loader= train_loader,
    #     test_loader= test_loader,
    #     criterion= loss,
    #     optimizer= unet_optimizer,
    #     num_epochs= 2,
    #     metric= metric,
    #     device= device,
    #     lr_scheduler= False
    # )
    
    # fpn_model = Model(
    #     model_name= 'Convolutional Feature Pyramid Network',
    #     model= fpn,
    #     train_loader= train_loader,
    #     test_loader= test_loader,
    #     criterion= loss,
    #     optimizer= fpn_optimizer,
    #     num_epochs= 25,
    #     metric= metric,
    #     device= device,
    #     lr_scheduler= False # True
    # )

    # resnext_model = Model(
    #     model_name= "ResNeXt50",
    #     model= rx50,
    #     train_loader= train_loader,
    #     test_loader= test_loader,
    #     criterion= loss,
    #     optimizer= rx50_optimizer,
    #     num_epochs= 25,
    #     metric= metric,
    #     device= device,
    #     lr_scheduler= False
    # )
    
    # # run 
    # unet_model.train_validate()
    # #unet_model.plotModel_dice()

    # #fpn_model.train_validate()
    # #fpn_model.plotModel_dice()
    # # rx50_lh, rx50_th, rx50_vh = train_model("ResNeXt50", rx50, train_loader, test_loader, bce_dice_loss, rx50_optimizer, False, 10)
    # #resnext_model.train_validate()
    # #resnext_model.plotModel_dice()
    pass

