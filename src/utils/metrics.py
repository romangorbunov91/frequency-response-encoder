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
        # Input shapes: [B, C, Length]
        
        predictions = torch.sigmoid(logits)
        targets = targets.to(predictions.dtype)

        # Dice coefficient.
        intersection = (predictions * targets).sum(dim=2)
        union = predictions.sum(dim=2) + targets.sum(dim=2)
        
        dice = (2. * intersection) / (union + self.eps)
        
        # Dice loss.
        return 1.0 - dice.mean()

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
    # Input shapes: [B, C, Length]
    
    predictions = (torch.sigmoid(logits) > threshold).float()
    targets = targets.to(predictions.dtype)
    
    intersection = (predictions * targets).sum(dim=2)
    union = predictions.sum(dim=2) + targets.sum(dim=2)
    
    dice = (2. * intersection) / (union + eps)
    
    return dice.mean().item()

@torch.no_grad()
def iou_score(logits, targets, threshold=0.5, eps=1e-6):
    # Input shapes: [B, C, Length]
    
    predictions = (torch.sigmoid(logits) > threshold).float()
    targets = targets.to(predictions.dtype)

    intersection = (predictions * targets).sum(dim=2)
    union = predictions.sum(dim=2) + targets.sum(dim=2)
    
    iou = intersection / (union - intersection + eps)
    
    return iou.mean().item()

@torch.no_grad()
def pixel_accuracy(logits, targets, threshold=0.5):
    # Input shapes: [B, C, Length]
    
    predictions = (torch.sigmoid(logits) > threshold).float()
    targets = targets.to(predictions.dtype)
    
    correct = (predictions == targets).float().sum(dim=2)
    
    return (correct / predictions.shape[2]).mean().item()