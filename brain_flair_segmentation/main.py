from __future__ import annotations
from train import * 



    

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
    '''
    fpn = smp.FPN(
        encoder_name= "efficientnet-b7",
        encoder_weights= "imagenet",
        in_channels= 3,
        classes= 1,
        activation= 'sigmoid',
    )
    fpn.to(device)
    '''
    
    fpn = FPN().to(device)
    output = fpn(torch.randn(1,3,256,256).to(device))
    print(output.shape)
    
    
    
    criterion = BCE_DiceCriterion()
    
    optimizer = torch.optim.AdamW(fpn.parameters(), lr= 1e-3)
    iou_metric = Iou_Metric()    
    dice_metric = Dice_Metric()
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer= optimizer, 
        patience= 2, 
        factor= 0.2
    )
    
    model_fpn = Model(
        50, 
        fpn, 
        train_loader, 
        test_loader, 
        optimizer, 
        criterion, 
        iou_metric, 
        dice_metric,
        scheduler,
        device 
    )
    model_fpn.train_validate()
    
   