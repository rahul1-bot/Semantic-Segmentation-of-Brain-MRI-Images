from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


#@: Custom Metric Class: 
class Metric(nn.Module):
    def __init__(self) -> None:
        super(Metric, self).__init__()
    

    def forward(self, inputs: np.ndarray, target: np.ndarray) -> float:
        intersection = 2.0 * (target * inputs).sum()
        union = target.sum() + inputs.sum()
        if target.sum() == 0 and inputs.sum() == 0:
            return 1.0
        return intersection / union




#@: Custom Loss Class 
class Loss(nn.Module):
    def __init__(self) -> None:
        super(Loss, self).__init__()
    

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return Loss._bce_dice_loss(inputs, target)
    

    @staticmethod
    def _dice_coeff_loss(inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        smooth: float = 1.0
        intersection = 2.0 * ((target * inputs).sum()) + smooth
        union = target.sum() + inputs.sum() + smooth
        return 1 - (intersection / union)
    
    
    @staticmethod
    def _bce_dice_loss(inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice_score = Loss._dice_coeff_loss(inputs, target)
        bce_score = nn.BCELoss()
        bce_loss = bce_score(inputs, target)
        return bce_loss + dice_score



#@: Model Adaptor Class 
class Model():
    def __init__(self, model_name: str, model: object, 
                                        train_loader: object, 
                                        test_loader: object,
                                        criterion: object,
                                        optimizer: object,
                                        num_epochs: int,
                                        metric: object,
                                        device: torch.device, 
                                        lr_scheduler: Optional[bool] = False) -> None:
        super(Model, self).__init__()
        self.model_name = model_name
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.metric = metric
        self.device = device
        self.lr_scheduler = lr_scheduler



    def train_validate(self, history: Optional[bool] = False) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        print(self.model_name)
        self.loss_history: list = []
        self.train_history: list = []
        self.val_history: list = []

        for epoch in range(self.num_epochs):
            self.model.train()
            losses: list = []
            train_iou: list = []
                    
            if self.lr_scheduler:
                warmup_factor: float = 1.0 / 100
                warmup_iters: int = min(100, len(self.train_loader) - 1)
                self.lr_scheduler = Model._warmup_lr_scheduler(warmup_iters, warmup_factor)
            
            for i_step, (data, target) in tqdm(enumerate(self.train_loader)):
                data = data.to(self.device)
                target = target.to(self.device)
                outputs = self.model(data)
                
                out_cut = np.copy(outputs.data.cpu().numpy())
                out_cut[np.nonzero(out_cut < 0.5)] = 0.0
                out_cut[np.nonzero(out_cut >= 0.5)] = 1.0
                
                train_dice = self.metric(out_cut, target.data.cpu().numpy())
                loss = self.criterion(outputs, target)
                
                losses.append(loss.item())
                train_iou.append(train_dice)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
                if self.lr_scheduler:
                    self.lr_scheduler.step()

            torch.save(self.model.state_dict(), f'{self.model_name}_{str(epoch)}_epoch.pt')
            val_mean_iou = self._compute_iou()
            
            self.loss_history.append(np.array(losses).mean())
            self.train_history.append(np.array(train_iou).mean())
            self.val_history.append(val_mean_iou)
            
            print("Epoch [%d]" % (epoch))
            print("Mean loss on train:", np.array(losses).mean(), 
                  "\nMean DICE on train:", np.array(train_iou).mean(), 
                  "\nMean DICE on validation:", val_mean_iou
            )
        
        if history:
            return self.loss_history, self.train_history, self.val_history
    


    def _compute_iou(self, threshold: Optional[float] = 0.3) -> float:
        self.model.eval()
        valloss: float = 0
        with torch.no_grad():
            for i_step, (data, target) in enumerate(self.test_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                outputs = self.model(data)
            
                out_cut = np.copy(outputs.data.cpu().numpy())
                out_cut[np.nonzero(out_cut < threshold)] = 0.0
                out_cut[np.nonzero(out_cut >= threshold)] = 1.0

                picloss = self.metric(out_cut, target.data.cpu().numpy())
                valloss += picloss

        return valloss / i_step



    @staticmethod
    def _warmup_lr_scheduler(warmup_iters: int, warmup_factor: float) -> torch.optim.lr_scheduler.LRScheduler():
        def f(x: int) -> int| float:
            if x >= warmup_iters:
                return 1
            alpha = float(x) / warmup_iters
            return warmup_factor * (1 - alpha) + alpha

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, f)



    def plotModel_dice(self) -> 'plot':
        x = np.arange(self.num_epochs)
        fig = plt.figure(figsize= (10, 6))
        plt.plot(x, self.train_history, label= 'train dice', lw= 3)
        plt.plot(x, self.val_history, label= 'validation dice', lw= 3)
        plt.title(f'{self.model_name}', fontsize= (10, 6))
        plt.legend(fontsize= 12)
        plt.xlabel('Epoch', fontsize= 15)
        plt.ylabel('Dice', fontsize= 15)
        plt.show()



#@: Driver Code
if __name__.__contains__('__main__'):
    pass