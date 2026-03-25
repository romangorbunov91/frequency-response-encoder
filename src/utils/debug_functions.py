import numpy as np
import os
from pathlib import Path
from typing import Union

import torch

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def visualize_predictions(
    logits_all,
    masks_all,
    dice_func,
    iou_func,
    threshold: float=0.5,
    columns: int=1,
    save_path: Union[str, Path] = None
    ):
    
    predictions_all = (torch.sigmoid(logits_all) > threshold).float().cpu()
    masks_all = masks_all.cpu()
    
    num_samples = len(masks_all)
    
    fig, axes = plt.subplots(num_samples//columns, columns, figsize=(8*columns, 2*num_samples/columns), constrained_layout=True)
    axes = axes.flatten()
       
    COLOR_OVERLAP = np.array([0.56, 0.79, 0.30])
    COLOR_TRUE_ONLY = np.array([0.80, 0.33, 0.20])
    COLOR_PRED_ONLY = np.array([0.35, 0.35, 0.35])
    
    for idx, masks, predictions, logits in zip(range(num_samples), masks_all, predictions_all, logits_all.cpu()):
        mask_true = np.asarray(masks, dtype=bool)
        mask_pred = np.asarray(predictions, dtype=bool)
        
        mask_overlap = mask_true & mask_pred      
        mask_true_only = mask_true & (~mask_pred)
        mask_pred_only = (~mask_true) & mask_pred
       
        h, w = mask_true.shape
        # Start with white background.
        mask_rgb = np.ones((h, w, 3))
        
        mask_rgb[mask_overlap] = COLOR_OVERLAP
        mask_rgb[mask_true_only] = COLOR_TRUE_ONLY
        mask_rgb[mask_pred_only] = COLOR_PRED_ONLY
        
        axes[idx].imshow(mask_rgb, aspect='auto', interpolation='nearest')
        axes[idx].set_title(
            f'Sample {idx}: '
            f'Dice={dice_func(logits.unsqueeze(0), masks.unsqueeze(0)):.4f}, '
            f'IoU={iou_func(logits.unsqueeze(0), masks.unsqueeze(0)):.4f}',
            fontsize=10
        )
        
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_OVERLAP, edgecolor='black', label='Overlap'),
        mpatches.Patch(facecolor=COLOR_TRUE_ONLY, edgecolor='black', label='True Mask'),
        mpatches.Patch(facecolor=COLOR_PRED_ONLY, edgecolor='black', label='Predicted Mask')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05), 
               ncol=len(legend_elements), fontsize=10, frameon=True)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"Test scores saved to {save_path}")
    plt.close(fig)