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

class PerChannelBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(PerChannelBCEWithLogitsLoss, self).__init__()
        # Set reduction='none' to handle averaging manually.
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Input shapes: [B, C, Length]
        
        # 1. Calculate loss element-wise (no reduction yet).
        # loss shape: [B, C, Length].
        loss = self.bce(logits, targets)
        
        # 2. Average over spatial dimensions (Length) first.
        # This gives us a loss value per Channel per Batch item
        # loss shape: [B, C].
        loss = loss.mean(dim=2) 
        
        # 3. Average over Channels and Batch.
        # This ensures each channel contributes equally to the final scalar loss
        return loss.mean()

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

        B, C = logits.shape[0], logits.shape[1]
        predictions = predictions.reshape(B, C, -1)
        targets = targets.reshape(B, C, -1)

        # Dice coefficient.
        intersection = (predictions * targets).sum(dim=2)
        union = predictions.sum(dim=2) + targets.sum(dim=2)
        dice = (2. * intersection + self.eps) / (union + self.eps)
        
        # Dice loss.
        return 1.0 - dice.mean()

class CombinedLoss(nn.Module):

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = PerChannelBCEWithLogitsLoss()
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
    
    B, C = logits.shape[0], logits.shape[1]
    predictions = predictions.reshape(B, C, -1)
    targets = targets.reshape(B, C, -1)
    
    intersection = (predictions * targets).sum(dim=2)
    dice = (2. * intersection + eps) / (predictions.sum(dim=2) + targets.sum(dim=2) + eps)
    
    return dice.mean().item()

@torch.no_grad()
def iou_score(logits, targets, threshold=0.5, eps=1e-6):
    # Input shapes: [B, C, Length]
    
    predictions = (torch.sigmoid(logits) > threshold).float()
    targets = targets.to(predictions.dtype)
    
    B, C = logits.shape[0], logits.shape[1]
    predictions = predictions.reshape(B, C, -1)
    targets = targets.reshape(B, C, -1)
    
    intersection = (predictions * targets).sum(dim=2)
    union = predictions.sum(dim=2) + targets.sum(dim=2) - intersection
    
    iou = (intersection + eps) / (union + eps)
    
    return iou.mean().item()

@torch.no_grad()
def pixel_accuracy(logits, targets, threshold=0.5):
    # Input shapes: [B, C, Length]
    
    predictions = (torch.sigmoid(logits) > threshold).float()
    targets = targets.to(predictions.dtype)
    
    B, C = logits.shape[0], logits.shape[1]
    predictions = predictions.reshape(B, C, -1)
    targets = targets.reshape(B, C, -1)
    
    correct = (predictions == targets).float().sum(dim=2)
    
    return (correct / predictions.shape[2]).mean().item()