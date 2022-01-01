from __future__ import annotations
from loss_metric import * 
from typing import Union, Any, Optional


class Early_Stop(nn.Module):
    def __init__(self, patience: Optional[int] = 6, min_delta: Optional[int] = 0, 
                                                    model_weight: Optional[str] = 'model_weights.pt') -> None:
        super(Early_Stop, self).__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.model_weight = model_weight
        self.counter: int = 0
        self.best_loss: float = float('inf')
        
    
    
    def load_weights(self, model: nn.Module) -> nn.Module:
        return model.load_state_dict(torch.load(self.model_weight))
    
    
    
    def forward(self, test_loss: float, model: nn.Module) -> bool:
        if self.best_loss - test_loss > self.min_delta:
            self.best_loss = test_loss
            torch.save(model.state_dict(), self.model_weight)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        
        return False
    
    
    
    

class Model(nn.Module):
    def __init__(self, epochs: int, model: nn.Module, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader, 
                                                                                                 optimizer: torch.optim, 
                                                                                                 criterion: BCE_DiceCriterion,
                                                                                                 iou_metric: Iou_Metric,
                                                                                                 dice_metric: Dice_Metric,
                                                                                                 scheduler: torch.optim.lr_scheduler,
                                                                                                 device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')) -> None:
        super(Model, self).__init__()
        self.epochs = epochs
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.iou_metric = iou_metric
        self.dice_metric = dice_metric
        self.scheduler = scheduler
        self.device = device
    
    
    
    def train_validate(self) -> dict[str, Union[Any, float]]:
        self.history: dict[str, list] = {
            x: [] for x in ['train_loss', 'test_loss', 'test_Iou', 'test_dice']
        }
        early_stopping = Early_Stop(patience = 7)
        
        for epoch in range(1, self.epochs + 1):
            start_time: float = time.time()
            running_loss: float = 0.0
            self.model.train()
            
            for idx, data in enumerate(tqdm(self.train_loader)):
                image, mask = data
                image, mask = image.to(self.device), mask.to(self.device)
                predictions = self.model(image)
                predictions = predictions.squeeze(1)
                loss = self.criterion(predictions, mask)
                running_loss += loss.item() * image.size(0)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            
            self.model.eval()
            
            with torch.no_grad():
                running_iou: float = 0.0
                running_dice: float = 0.0
                running_test_loss: float = 0.0
                for idx, data in enumerate(tqdm(self.test_loader)):
                    image, mask = data
                    image, mask = image.to(self.device, dtype= torch.float), mask.to(self.device, dtype= torch.float)
                    predictions = self.model(image)
                    predictions = predictions.squeeze(1)
                    running_dice += self.dice_metric(predictions, mask).sum().item()
                    running_iou += self.iou_metric(predictions, mask).sum().item()
                    loss = self.criterion(predictions, mask)
                    running_test_loss += loss.item() * image.size(0)
            
            train_loss = running_loss/ len(self.train_loader.dataset)
            test_loss = running_test_loss/ len(self.test_loader.dataset)
            test_dice = running_dice/ len(self.test_loader.dataset)
            test_iou = running_iou/ len(self.test_loader.dataset)
            
            self.history['train_loss'].append(train_loss)
            self.history['test_loss'].append(test_loss)
            self.history['test_Iou'].append(test_iou)
            self.history['test_dice'].append(test_dice)
            
            data_map: dict[str, Any] = {
                'Epoch': f'{epoch}/ {self.epochs}',
                'Training Loss': train_loss,
                'Testing Loss': test_loss,
                'Testing Mean IOU': test_iou,
                'Testing Dice Coefficient': test_dice
            }
            print(data_map)
            
            self.scheduler.step(test_loss)
            if early_stopping(test_loss, self.model):
                early_stopping.load_weights(self.model)
                break
            
        self.model.eval()
        return self.history
    
    
    
    
    def valLoss(self) -> 'plot':
        train_loss_list: list[float] = [
            float(x) for x in self.history['train_loss']
        ]
        test_loss_list: list[float] =  [
            float(x) for x in self.history['test_loss']
        ]
        plt.plot(train_loss_list, '-bx')
        plt.plot(test_loss_list, '-rx')
        plt.title('Model Loss Plot')
        plt.xlabel('No of Epoch')
        plt.ylabel('Loss')
        plt.legend(['train', 'validation'], loc= 'best')
        plt.show()
    
    
    
    def test_iou_dice(self) -> 'plot':
        test_iou: list[float] = [
            float(x) for x in self.history['test_Iou']
        ]
        test_dice: list[float] = [
            float(x) for x in self.history['test_dice']
        ]
        plt.plot(test_iou, '-bx')
        plt.plot(test_dice, '-rx')
        plt.title('Model IoU and Dice Score')
        plt.xlabel('No of Epochs')
        plt.ylabel('Score')
        plt.legend(['IoU', 'Dice Score'], loc= 'best')
        plt.show()
    

    
        
        