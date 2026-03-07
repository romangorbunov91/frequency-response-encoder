import torch
import torch.nn as nn

class AverageMeter(object):
    """Average Meter object, contain val, avg, sum and count on concurrent values"""
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class DiceLoss(nn.Module):
    """   
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    """
    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        
        predictions = torch.sigmoid(logits)
        targets = targets.to(predictions.dtype)
        
        # Flatten.
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Dice coefficient.
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.eps) / (predictions.sum() + targets.sum() + self.eps)
        
        # Dice loss.
        return 1.0 - dice

class CombinedLoss(nn.Module):

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

@torch.no_grad()
def dice_coefficient(logits, targets, threshold=0.5, eps=1e-6):
    
    predictions = (torch.sigmoid(logits) > threshold).float()
    
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    intersection = (predictions * targets).sum()
    dice = (2. * intersection + eps) / (predictions.sum() + targets.sum() + eps)
    
    return dice.item()

@torch.no_grad()
def iou_score(logits, targets, threshold=0.5, eps=1e-6):
    
    predictions = (torch.sigmoid(logits) > threshold).float()
    
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection
    
    iou = (intersection + eps) / (union + eps)
    
    return iou.item()

@torch.no_grad()
def pixel_accuracy(logits, targets, threshold=0.5):
  
    predictions = (torch.sigmoid(logits) > threshold).float()
    correct = (predictions == targets).float().sum()
    
    return (correct / targets.numel()).item()