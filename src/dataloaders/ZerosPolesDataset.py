import numpy as np
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Union, Tuple, List, Optional

def positions_to_mask(
    positions,
    total_bits: int,
    halfwindow: int=0
    ):
    
    """Convert list of bit positions to an integer mask."""
    
    mask = [0] * total_bits
    for pos in positions:

        start = max(0, pos - halfwindow)
        end = min(total_bits, pos + halfwindow + 1)

        mask[start:end] = [1] * (end - start)
    
    return mask


@dataclass
class TransformsConfig:
    crop_ratio: List[float] = field(default_factory=lambda: [1.0, 1.0])
    time_delay: List[float] = field(default_factory=lambda: [0.0, 0.0])
    noise_level: List[float] = field(default_factory=lambda: [0.0, 0.0])
    noise_reduce: int = 0
    gain: List[float] = field(default_factory=lambda: [1.0, 1.0])
    

class ZerosPolesDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Union[str, Path],
        split: str,
        mask_halfwindow: int = 0,
        samples: Optional[List] = None,
        transforms: Optional[TransformsConfig] = None
    ):
        
        super().__init__()
        
        self.dataset_path = Path(dataset_dir) / split
        
        mask_path = Path(dataset_dir) / (split + "_masks.json")
        assert mask_path.exists(), f"Mask not found: {mask_path}"
        with open(mask_path, "r") as f:
            self.masks = json.load(f)
        
        self.mask_halfwindow = mask_halfwindow
        
        # Read all samples.
        if samples is None:
            self.samples = list(self.masks.keys())
        else:
            self.samples = samples
        
        self.transforms = transforms
            
    def __len__(self) -> int:
        return len(self.samples)

    def _augmentations_(self, data_tensor, masks_tensor):
        
        crop_ratio = self.transforms.crop_ratio
        time_delay = self.transforms.time_delay
        noise_level = self.transforms.noise_level
        noise_reduce = self.transforms.noise_reduce
        gain = self.transforms.gain
        
        # 1. Crop-Resize Augmentation: both data and masks.
        if min(crop_ratio) < 1.0:

            N = data_tensor.shape[-1]          
            
            # Determine random crop length.
            N_crop = int((crop_ratio[0] + torch.rand(1).item() * (crop_ratio[1] - crop_ratio[0])) * N)

            # Determine random start index.
            start_idx = 0
            if N_crop < N:
                start_idx = torch.randint(0, N - N_crop + 1, (1,)).item()
            
            # Slice tensors (keep dimensions for interpolation).
            data_crop = data_tensor[:, start_idx:(start_idx + N_crop)]
            masks_crop = masks_tensor[:, start_idx:(start_idx + N_crop)]
            
            # Add batch dimension since torch.nn.functional.interpolate expects (Batch, Channel, Length).
            # Then remove back.
            data_tensor = F.interpolate(data_crop.unsqueeze(0), size=N, mode='linear', align_corners=False).squeeze(0)
            
            masks_tensor_remaped = torch.zeros_like(masks_tensor)
            ch_idxs, idxs = torch.where(masks_crop > 0.5)
            if idxs.numel() > 0:
                # Map coordinates from N_crop space to N space.
                new_idxs = (idxs.float() * N / N_crop).round().long()#.clamp(0, N-1)
                masks_tensor_remaped[ch_idxs, new_idxs] = 1
                
            masks_tensor = masks_tensor_remaped

        freq_tensor = data_tensor[0 ,:]
        data_tensor = data_tensor[1:,:]
        
        # 2. Random time-delay (Data only).
        if max(time_delay) > 0.0:
            omega_delay = -2*np.pi * freq_tensor * (time_delay[0] + torch.rand(1).item() * (time_delay[1] - time_delay[0]))

            # Precompute trigonometric values
            cos_theta_tensor = torch.cos(omega_delay)
            sin_theta_tensor = torch.sin(omega_delay)

            # Extract real and imaginary components.
            real_tensor = data_tensor[0,:]
            imag_tensor = data_tensor[1,:]

            # Complex rotation.
            delay_real_tensor = real_tensor * cos_theta_tensor - imag_tensor * sin_theta_tensor
            delay_imag_tensor = real_tensor * sin_theta_tensor + imag_tensor * cos_theta_tensor

            # Reconstruct phase-shifted tensor.
            data_tensor = torch.stack([delay_real_tensor, delay_imag_tensor], dim=0)

        # 3. Random Noise Augmentation (Data only).
        if max(noise_level) > 0.0:
            noise = torch.randn_like(data_tensor)
            
            for _ in range(noise_reduce):
                noise *= torch.randint(0, 2, size=noise.shape, dtype=noise.dtype)
                       
            data_tensor += noise * data_tensor.std(dim=-1, keepdim=True) * (noise_level[0] + torch.rand(1).item() * (noise_level[1] - noise_level[0]))

        # 4. Random gain (Data only).
        data_tensor *= (gain[0] + torch.rand(1).item() * (gain[1] - gain[0]))
                    
        return data_tensor, masks_tensor, freq_tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        sample_id = self.samples[idx]
        sample_path = self.dataset_path / f"{sample_id}.csv"

        if not sample_path.exists():
            raise FileNotFoundError(f"File not found: {sample_path}")
        
        data_np = np.loadtxt(self.dataset_path / f"{sample_id}.csv", delimiter=',', skiprows=1)
        data_tensor = torch.from_numpy(data_np.T).float()
        
        mask_dict = self.masks[sample_id]
        masks_list = []
        for key, positions in mask_dict.items():
            if key == 'zero_poles':
                continue
            masks_list.append(
                positions_to_mask(
                    positions=positions,
                    total_bits=data_tensor.shape[-1],
                    halfwindow=self.mask_halfwindow
                    )
                )
        masks_tensor = torch.from_numpy(np.vstack(masks_list)).float()
        
        if self.transforms:
            data_tensor, masks_tensor, freq_tensor = self._augmentations_(data_tensor, masks_tensor)
        else:
            freq_tensor = data_tensor[0 ,:]
            data_tensor = data_tensor[1:,:]

        return data_tensor, masks_tensor, freq_tensor