from __future__ import annotations
from train import * 
from models.unet import UNet
from models.fpn import Feature_PyramidNetwork
from models.resnext_unet import ResNeXtUNet


__authors__: list[str] = [
    'Rahul Sawhney', 'Shubham Garg', 'Aabha Malik', 'Harsh Sharma'
]

__authors_email__: dict[str, str] = {
    'Rahul Sawhney': 'sawhney.rahulofficial@outlook.com',
    'Shubham Garg': 'shubgarg17@gmail.com',
    'Aabha Malik': 'aabhamalik30@gmail.com',
    'Harsh Sharma': 'sharma.harsh3107@gmail.com'
}

__authors_qualifications__: dict[str, str] = {
    x: 'Btech CSE, Amity University, Noida' 
    for x in ['Rahul Sawhney',  'Shubham Garg', 'Aabha Malik', 'Harsh Sharma']    

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
            Semantic Segmentation of brain MRI Images  
    
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
            used a hybrid model of UNet shaped ResNeXT model which consists of series of encoders and decoders.
            Along with this the AdamW are used as optimizer. Criterion used is IOU 
            (Intersection Over Union) and the metric utilized for evaluation is Dice Score. 
    
    >>> Paper Keywords: 
            Lower-grade gliomas detection, 
            Brain Semantic Segmentation, 
            Image Recogonition, 
            Feature-Detector-Network

'''


#@: Driver Code
if __name__.__contains__('__main__'):
    path: str = 'C:\\Users\\RAHUL\\OneDrive\\Desktop\\brain_segmentation\\kaggle_3m'
    
    transform_list = A.Compose([
        A.ChannelDropout(p=0.3),
        A.RandomBrightnessContrast(p=0.3)
    ])
    
    brain_data: object = BrainSegmentorData(path, transform_list)
    brain_metadata, brain_df = brain_data.get_data()
    
    for image, mask in brain_data:
        print(image.shape)
        print(mask.shape)
        break
    
    print(len(brain_data))
    train_data, test_data = torch.utils.data.random_split(brain_data, [3600, 329])
    
    train_loader = torch.utils.data.DataLoader(
        dataset= train_data, 
        batch_size= 12, 
        shuffle= True
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset= test_data, 
        batch_size= 12
    )
    
    dataloaders: dict[str, torch.utils.data.DataLoader] = {
        'train': train_loader,
        'test': test_loader
    }
    dataset_sizes: dict[str, int] = {
        'train': len(train_data),
        'test': len(test_data)
    }
    
    device: torch.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    unet = UNet(n_classes= 1).to(device)
    fpn = Feature_PyramidNetwork().to(device)
    resnext_unet = ResNeXtUNet(n_classes= 1).to(device)
    
    criterion = BCE_DiceCriterion()
    iou_metric = Iou_Metric()    
    dice_metric = Dice_Metric()
    
    
    unet_optimizer = torch.optim.AdamW(unet.parameters(), lr= 1e-3)
    fpn_optimizer = torch.optim.AdamW(fpn.parameters(), lr= 1e-3)
    resnext_unet_optimizer = torch.optim.AdamW(resnext_unet.parameters(), lr= 1e-3)
    
    
    unet_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer= unet_optimizer, 
        patience= 2, 
        factor= 0.2
    )
    fpn_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer= fpn_optimizer, 
        patience= 2, 
        factor= 0.2
    )
    resnext_unet_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer= resnext_unet_optimizer, 
        patience= 2, 
        factor= 0.2
    )
    
    model_unet = Model(
        epochs = 50,
        model = unet,
        train_loader = train_loader,
        test_loader = test_loader,
        optimizer = unet_optimizer,
        criterion = criterion,
        iou_metric = iou_metric,
        dice_metric = dice_metric,
        scheduler = unet_scheduler,
        device = device 
    )
    
    model_fpn = Model(
        epochs = 50,
        model = fpn,
        train_loader = train_loader,
        test_loader = test_loader,
        optimizer = fpn_optimizer,
        criterion = criterion,
        iou_metric = iou_metric,
        dice_metric = dice_metric,
        scheduler = fpn_scheduler,
        device = device 
    )
    
    model_resnext_unet = Model(
        epochs = 50,
        model = resnext_unet,
        train_loader = train_loader,
        test_loader = test_loader,
        optimizer = resnext_unet_optimizer,
        criterion = criterion,
        iou_metric = iou_metric,
        dice_metric = dice_metric,
        scheduler = resnext_unet_scheduler,
        device = device 
    )
    
    model_unet.train_validate()
    model_unet.valLoss()
    model_unet.test_iou_dice()
    
    model_fpn.train_validate()
    model_fpn.valLoss()
    model_fpn.test_iou_dice()
    
    model_resnext_unet.train_validate()
    model_resnext_unet.valLoss()
    model_resnext_unet.test_iou_dice()
    