from __future__ import annotations
from dataset import *



class Iou_Metric(nn.Module):
    def __init__(self) -> None:
        super(Iou_Metric, self).__init__()
    
    
    def forward(self, predictions: torch.Tensor, labels: torch.Tensor, 
                                                 smoothing: Optional[float] = 1e-7) -> torch.Tensor:
        predictions = torch.where(predictions > .5, 1, 0)
        labels = labels.byte()
        intersections: torch.Tensor = (predictions & labels).float().sum((1, 2))
        union: torch.Tensor = (predictions | labels).float().sum((1, 2))
        iou: torch.Tensor = (intersections + smoothing) / (union + smoothing)
        return iou
    
    

    
class Dice_Metric(nn.Module):
    def __init__(self) -> None:
        super(Dice_Metric, self).__init__()
        
    
    def forward(self, predictions: torch.Tensor, labels: torch.Tensor, smoothing: Optional[float] = 1e-7) -> torch.Tensor:
        predictions = torch.where(predictions > .5, 1, 0)
        labels = labels.byte()
        intersections: torch.Tensor = (predictions & labels).float().sum((1, 2))
        return (
            ((2 * intersections) + smoothing) / (predictions.float().sum((1, 2)) + labels.float().sum((1, 2)) + smoothing)
        )
        




class BCE_DiceCriterion(nn.Module):
    def __init__(self) -> None:
        super(BCE_DiceCriterion, self).__init__()
        
    
    def forward(self, output: torch.Tensor, target: torch.Tensor, alpha: Optional[float] = 0.01) -> torch.Tensor:
        bce: torch.Tensor = F.binary_cross_entropy(output, target)
        dice_coeff: Dice_Metric = Dice_Metric()
        soft_dice = 1 - dice_coeff(output, target).mean()
        return bce + alpha * soft_dice