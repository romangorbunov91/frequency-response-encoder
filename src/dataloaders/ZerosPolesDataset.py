import numpy as np
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Union, Tuple, List, Optional

def positions_to_mask(
    positions: List[int],
    total_bits: int,
    halfwindow: int=0
    ) -> List[int]:
    
    # Convert list of bit positions to an integer mask.
    
    mask = [0] * total_bits
    for pos in positions:

        start = max(0, pos - halfwindow)
        end = min(total_bits, pos + halfwindow + 1)

        mask[start:end] = [1] * (end - start)
    
    return mask


@dataclass
class TransformsConfig:
    gain: List[float]=field(default_factory=lambda: [1.0, 1.0])
    phase_delay: List[float]=field(default_factory=lambda: [0.0, 0.0])
    noise_level: List[float]=field(default_factory=lambda: [0.0, 0.0])
    noise_reduce: int=0

    def __post_init__(self):
        if any(x <= 0 for x in self.gain):
            raise ValueError("Gain values must be strictly positive for log10 scaling.")
        if self.gain[0] > self.gain[1]:
            raise ValueError("gain[0] must be <= gain[1].")
        if any(x < 0 for x in self.phase_delay):
            raise ValueError("Delay values can't be negative since "-" is applied inside.")
        if self.phase_delay[0] > self.phase_delay[1]:
            raise ValueError("phase_delay[0] must be <= phase_delay[1].")
        if self.noise_level[0] > self.noise_level[1]:
            raise ValueError("noise_level[0] must be <= noise_level[1].")
    

class GeneralTransforms:
    def __init__(self,
        config: Optional[TransformsConfig] = None,
        rng: Optional[np.random.Generator] = None
        ):
        
        self.config = config if config is not None else TransformsConfig()
        
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

    def __call__(self, data: np.ndarray) -> np.ndarray:
        
        # Gets and Returns np.ndarray of shape (3, Length) with dtype float:
        # 0 - frequency
        # 1 - magnitude
        # 2 - phase
        
        gain = self.config.gain
        phase_delay = self.config.phase_delay
        noise_level = self.config.noise_level
        noise_reduce = self.config.noise_reduce

        rng = self.rng
        
        freq = data[0, :]
        magnitude = data[1, :]
        phase = data[2, :]
        
        # Random gain (magnitude only).
        if any(x != 1.0 for x in gain):
            magnitude += 20*np.log10(gain[0] + rng.random() * (gain[1] - gain[0]))
        
        # Random phase-delay (phase only).
        if max(phase_delay) > 0.0:
            phase_coeff = (phase_delay[0] + rng.random() * (phase_delay[1] - phase_delay[0])) / freq[-1]
            phase -= freq * phase_coeff

        # Random Noise.
        
        # Combine before noising.
        mag_ph = np.vstack([magnitude, phase])
        
        if max(noise_level) > 0.0:
            noise = rng.standard_normal(size=mag_ph.shape)
            noise_mask = (rng.random(size=noise.shape) < (0.5 ** noise_reduce)).astype(noise.dtype)
            data_std = np.std(mag_ph, axis=-1, keepdims=True)
            scale = noise_level[0] + rng.random() * (noise_level[1] - noise_level[0])
            mag_ph += noise * noise_mask * data_std * scale
        
        return np.vstack([freq, mag_ph])

class ConversionTransforms:
    def __init__(self,
            num_iter: int=2,
            return_input: bool=False
            ):
        
        self.num_iter = num_iter
        self.return_input = return_input

    def __call__(self, data: np.ndarray) -> np.ndarray:
        
        # Skip frequencies (row=0).
        data_diff = data[1:,:].copy()
        if self.return_input:
            data_out = [data_diff]
        else:
            data_out = []

        for _ in range(self.num_iter):
            data_diff = np.gradient(data_diff, axis=1)
            data_out.append(data_diff)

        return np.vstack(data_out)
        

class ZerosPolesDataset(Dataset):
    def __init__(
            self,
            dataset_dir: Union[str, Path],
            split: str,
            mask_halfwindow: int = 0,
            samples: Optional[List] = None,
            transforms: Optional[List] = None
            ):
        
        super().__init__()
                
        mask_path = Path(dataset_dir) / (split + "_masks.json")
        assert mask_path.exists(), f"Mask not found: {mask_path}"
        with open(mask_path, "r") as f:
            self.masks = json.load(f)
        
        self.mask_halfwindow = mask_halfwindow
        
        if samples is None:
            # Read all samples.
            self.samples = list(self.masks.keys())
        else:
            self.samples = samples
        
        # Load full dataset to in-memory cache to avoid repeated np.loadtxt calls.
        self._sample_cache: dict[str, np.ndarray] = {}
        dataset_path = Path(dataset_dir) / split
        for sample_id in self.samples:
            sample_path = dataset_path / f"{sample_id}.csv"
            if not sample_path.exists():
                raise FileNotFoundError(f"File not found: {sample_path}")
            self._sample_cache[sample_id] = np.loadtxt(sample_path, delimiter=',', skiprows=1).T

        if transforms is None:
            self.transforms = []
        else:
            self.transforms = transforms

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        sample_id = self.samples[idx]

        # Load data from cache.
        data = self._sample_cache[sample_id].copy()
        
        freq = data[0, :]

        mask_dict = self.masks[sample_id]
        masks_list = []
        for key, positions in mask_dict.items():
            if key == 'zero_poles':
                continue
            masks_list.append(
                positions_to_mask(
                    positions=positions,
                    total_bits=len(freq),
                    halfwindow=self.mask_halfwindow
                    ),
                )
        masks = np.vstack(masks_list)

        # Augmentations and conversion to diff-type.
        for t in self.transforms:
            data = t(data)

        # Convert numpy to tensor.
        data_tensor = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))
        masks_tensor = torch.from_numpy(np.ascontiguousarray(masks, dtype=np.float32))
        freq_tensor = torch.from_numpy(np.ascontiguousarray(freq, dtype=np.float32))
        
        return data_tensor, masks_tensor, freq_tensor