import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, eps: float=1e-6):
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
        
        # Set Dice to 1.0 for channels with no ground truth
        empty_targets = targets.sum(dim=2) == 0
        dice = torch.where(empty_targets, torch.ones_like(dice), dice)
        
        return 1.0 - dice.mean()

class CombinedLoss(nn.Module):

    def __init__(self,
            bce_weight: float = 0.5,
            dice_weight: float = 0.5,
            ds_weights: list[float] | None = None
        ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.ds_weights = ds_weights if ds_weights is not None else [0.25, 0.5]
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dtype != torch.float32:
            targets = targets.float()

        if isinstance(logits, tuple):
            main_logits, ds_logits = logits
            all_logits = ds_logits + [main_logits]
            all_weights = self.ds_weights + [1.0]  # Main prediction always gets 1.0.
        else:
            all_logits = [logits]
            all_weights = [1.0]

        total_loss = 0.0
        target_len = targets.shape[-1]

        for d, w in zip(all_logits, all_weights):
            if d.shape[-1] != target_len:
                d_resized = F.interpolate(d, size=target_len, mode='nearest')
            else:
                d_resized = d
            
            bce_loss = self.bce(d_resized, targets)
            dice_loss = self.dice(d_resized, targets)
            total_loss += w * (self.bce_weight * bce_loss + self.dice_weight * dice_loss)
            
        return total_loss / sum(all_weights)


@torch.no_grad()
def dice_coefficient(logits: torch.Tensor, targets: torch.Tensor, threshold: float=0.5, eps: float=1e-6):
    # Input shapes: [B, C, Length]
    
    predictions = (torch.sigmoid(logits) > threshold).float()
    targets = targets.to(predictions.dtype)
    
    intersection = (predictions * targets).sum(dim=2)
    union = predictions.sum(dim=2) + targets.sum(dim=2)
    
    dice = (2. * intersection) / (union + eps)

    # Set Dice to 1.0 for channels with no ground truth.
    empty_targets = targets.sum(dim=2) == 0
    dice = torch.where(empty_targets, torch.ones_like(dice), dice)
    
    return dice.mean().item()

@torch.no_grad()
def iou_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float=0.5, eps: float=1e-6):
    # Input shapes: [B, C, Length]
    
    predictions = (torch.sigmoid(logits) > threshold).float()
    targets = targets.to(predictions.dtype)

    intersection = (predictions * targets).sum(dim=2)
    union = predictions.sum(dim=2) + targets.sum(dim=2)
    
    iou = intersection / (union - intersection + eps)
    
    # Set iou to 1.0 for channels with no ground truth.
    empty_targets = targets.sum(dim=2) == 0
    iou = torch.where(empty_targets, torch.ones_like(iou), iou)

    return iou.mean().item()

@torch.no_grad()
def pixel_accuracy(logits: torch.Tensor, targets: torch.Tensor, threshold: float=0.5, eps: float=1e-6):
    # Input shapes: [B, C, Length]
    
    predictions = (torch.sigmoid(logits) > threshold).float()
    targets = targets.to(predictions.dtype)
    
    correct = (predictions == targets).float().sum(dim=2)
    
    return (correct / (predictions.shape[2] + eps)).mean().item()