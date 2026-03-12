import os
from pathlib import Path

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms_v2
from torchvision.transforms.v2 import InterpolationMode

# Import Utils.
from torch.utils.data import DataLoader
from .utils.metrics import AverageMeter, CombinedLoss, dice_coefficient, iou_score, pixel_accuracy

# Import Datasets.
from .datasets.ZerosPolesDataset import ZerosPolesDataset

# Import Model.
from .models.model_utilizer import load_net, update_optimizer, ModelUtilizer
from .models.base_model import base_model

# Setting seeds.
def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % 2 ** 32)

class MetricsHistory:
    def initialize_metrics(self, metric_names, phases):
        """Call this in __init__ of your main class."""
        
        phase_keys = [phase.lower() for phase in phases]
        self.metrics = {
            name: {phase: AverageMeter() for phase in phase_keys}
            for name in metric_names
        }

        history_keys = [
            f"{split}_{name}"
            for split in phase_keys
            for name in list(self.metrics.keys())
            ]
        
        self.train_history = {key: [] for key in history_keys}
        self.train_history["epoch"] = []
        self.train_history["lr"] = []
        self.train_history["encoder_lr"] = []
    
    def print_metrics(self, phases):
        phase_keys = [phase.lower() for phase in phases]
        
        if len(self.train_history['epoch']) > 0:
            prefix = f"Epoch {self.train_history['epoch'][-1]:2d} | "
        else:
            prefix = ''
        
        if 'train' in phase_keys:
            train_str = ", ".join(f"{name}: {self.train_history[f'train_{name}'][-1]:.4f}" 
                                for name in list(self.metrics.keys()))
            print(f"{prefix}Train. {train_str}")
        
        if 'val' in phase_keys:
            val_str = ", ".join(f"{name}: {self.train_history[f'val_{name}'][-1]:.4f}" 
                                for name in list(self.metrics.keys()))
            print(f"{' ' * len(prefix)}Valid. {val_str}")
    
    def log_epoch_history(self, phases):
        
        if hasattr(self, 'epoch'):
            self.train_history["epoch"].append(self.epoch + 1)
        
        if hasattr(self, 'optimizer'):
            self.train_history["lr"].append(self.optimizer.param_groups[0]["lr"])
            if len(self.optimizer.param_groups) > 1:
                self.train_history["encoder_lr"].append(self.optimizer.param_groups[1]["lr"])
            else:
                self.train_history["encoder_lr"].append(None) 
        
        phase_keys = [phase.lower() for phase in phases]
        for name in self.metrics:
            for split in phase_keys:
                key = f"{split}_{name}"
                self.train_history[key].append(self.metrics[name][split].avg)

    def update_metrics(self, split: str, batch_size: int, **metrics: float) -> None:
        """
        Update metrics for a given split.
        
        Args:
            split (str): 'train' or 'val'.
            batch_size (int): batch size (used for weighted averaging).
            **metrics: keyword arguments like loss=..., accuracy=..., dice=..., iou=...
        """
        for name, value in metrics.items():
            if name in self.metrics:
                self.metrics[name][split].update(value, batch_size)
            else:
                raise KeyError(f"Metric '{name}' is not registered in self.metrics.")

    def reset_metrics(self):
        for metric_dict in self.metrics.values():
            for meter in metric_dict.values():
                meter.reset()

class ModelTrainer(MetricsHistory):

    def __init__(self, configer):
        self.configer = configer

        #: str: Type of dataset.
        self.dataset = self.configer.get("dataset_name").lower()
        self.data_path = Path(self.configer.general_config.get("data_dir")) / (self.configer.get("dataset_name") + '/images')
        
        # DataLoaders.
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # Module load and save utility.
        self.device = torch.device(self.configer.device)
        print(f"Device (train.py): {self.device}")
        self.model_utility = ModelUtilizer(self.configer)
        self.net = None

        # Training procedure.
        self.epoch = None
        self.epoch_init = None
        self.encoder_start_epoch = None
        self.optimizer = None
        self.scheduler = None
        self.loss = None
        
        # Augmentation.
        self.train_transforms.transforms_flag=False,
        self.train_transforms.crop_ratio=[0.8, 1.0],
        self.train_transforms.time_delay=[0.0, 1e-9],
        self.train_transforms.noise_level=[5e-3, 30e-3],
        self.train_transforms.noise_reduce3,
        self.train_transforms.gain=[-1e3, 1e3]
        
        # Same transforms for TRAIN, VAL, TEST.
        self.val_transforms = self.train_transforms
        self.test_transforms = self.train_transforms
        
        self.initialize_metrics(
            ['loss', 'dice', 'iou', 'accuracy'],
            ['train', 'val', 'test']
            )
        
    def init_model(self):
        """Initialize model and other data for procedure"""
        
        self.loss_func = CombinedLoss(bce_weight=0.5, dice_weight=0.5).to(self.device)
        
        mdl_input_size = self.configer.model_config.get('input_size')

        self.net = base_model(
            in_channels = mdl_input_size[0],
            out_channels = 1,
            features = self.configer.model_config['feature_list'],
            backbone_layers_config = self.configer.model_config['backbone_layers_num']*[self.configer.model_config['backbone_block_size']],
            backbone_layer0_channels = self.configer.model_config['feature_list'][0],
            device = self.device
            )

        # Initializing training.
        self.net, self.epoch_init, optim_dict, sched_dict = load_net(
            net = self.net,
            checkpoints_file = self.configer.get('resume'),
            device = self.device
            )
        self.epoch = self.epoch_init
        self.encoder_start_epoch = self.configer.model_config['epochs'] - self.configer.model_config['encoder_tune_epochs']
        
        # Set optimizer.
        self.optimizer = update_optimizer(
            net = self.net,
            optim = self.configer.model_config.get('solver_type'),
            lr = self.configer.model_config.get('base_lr'),
            decay = self.configer.model_config.get('weight_decay'),
            encoder_lr = self.configer.model_config.get('encoder_base_lr')
            )
        
        if optim_dict is None:
            print(f"Starting training {self.configer.model_config.get('model_name')} from scratch using {self.configer.model_config.get('solver_type')}.")
        else:
            self.optimizer.load_state_dict(optim_dict)
            print(f"Resuming training {self.configer.model_config.get('model_name')} from epoch {self.epoch} using {self.configer.model_config.get('solver_type')}.")
        
        if bool(self.configer.model_config['scheduler_on']):
            # Set scheduler.
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='max',
                factor=0.9,
                patience=self.configer.model_config['scheduler_patience']
            )
            if sched_dict is not None:
                self.scheduler.load_state_dict(sched_dict)
            print("Scheduler ON")
        
        self.model_size = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print(f"Model parameters: {self.model_size}")
        
        # Selecting Dataset and DataLoader.
        if self.dataset == "zeros-poles-dataset":
            self.train_geometric_augmentations = transforms_v2.Compose([
                transforms_v2.Resize(
                    size=tuple(mdl_input_size[-2:]),
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True
                ),
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.RandomVerticalFlip(p=0.5),
                transforms_v2.RandomRotation(degrees=45),
                transforms_v2.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0
                ),
            ])
            
            self.train_photometric_augmentations = [
                transforms_v2.ColorJitter(brightness=0.2, contrast=0.2),
                transforms_v2.GaussianNoise(mean=0.0, sigma=0.1),
                transforms_v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            ]

            self.val_geometric_augmentations = transforms_v2.Compose([
                transforms_v2.Resize(
                    size=tuple(mdl_input_size[-2:]),
                    interpolation=InterpolationMode.NEAREST,
                    antialias=True
                ),
            ])
            
            self.val_photometric_augmentations = None
            
            self.postprocessing = transforms_v2.Compose([
                transforms_v2.Normalize(mean=mean_norm, std=std_norm),
            ])
            
        else:
            raise NotImplementedError(f"Dataset not supported: {self.dataset}")

        # Setting Dataloaders.
        if self.dataset == "zeros-poles-dataset":
            img_prefix = self.configer.dataset_config.get('img_prefix')
            mask_prefix = self.configer.dataset_config.get('mask_prefix')
            img_folder = img_prefix + '/'
            all_images = [img_no_ext.replace(img_prefix, '') for img_no_ext in
                [img.replace('.png', '') for img in sorted(os.listdir(self.data_path / img_folder)) if img.endswith('.png')]
            ]

            
            self.train_loader = DataLoader(
                ZerosPolesDataset(
                    dataset_dir = self.dataset_dir,
                    split = 'train',
                    samples = ,
                    transforms_flag = True,
                    crop_ratio=[0.8, 1.0],
                    time_delay=[0.0, 1e-9],
                    noise_level=[5e-3, 30e-3],
                    noise_reduce=3,
                    gain=[-1e3, 1e3]
                    ), 
                batch_size=self.configer.model_config["batch_size"],
                shuffle=True,
                num_workers=self.configer.model_config["workers"],
                worker_init_fn=worker_init_fn,
                pin_memory=True)

            self.val_loader = DataLoader(
                ZerosPolesDataset(
                    dataset_dir = self.dataset_dir,
                    split = 'val',
                    samples = ,
                    transforms_flag = True,
                    crop_ratio=[0.8, 1.0],
                    time_delay=[0.0, 1e-9],
                    noise_level=[5e-3, 30e-3],
                    noise_reduce=3,
                    gain=[-1e3, 1e3]
                    ), 
                batch_size=self.configer.model_config["batch_size"],
                shuffle=False,
                num_workers=self.configer.model_config["workers"],
                pin_memory=True)
        else:
            raise NotImplementedError(f"Dataset not supported: {self.dataset}")
        
        print(f"TRAIN size: {self.train_loader.__len__}")
        print(f"VAL   size: {self.val_loader.__len__}")
        print(f"TEST  size: {self.test_loader.__len__}")
              
    def __train(self):
        """Train function for every epoch."""
        self.net.train()
        
        for data_tuple in tqdm(self.train_loader, desc="Train"):

            inputs, masks = data_tuple[0].to(self.device), data_tuple[1].to(self.device)
            
            outputs = self.net(inputs)

            self.optimizer.zero_grad()
            loss = self.loss_func(outputs, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1)
            self.optimizer.step()

            self.update_metrics(
                split = "train",
                batch_size = inputs.size(0),
                loss = loss.item(),
                dice = dice_coefficient(outputs.detach(), masks.detach()),
                iou = iou_score(outputs.detach(), masks.detach()),
                accuracy = pixel_accuracy(outputs.detach(), masks.detach()))

    def __val(self):
        """Validation function."""
        self.net.eval()

        with torch.no_grad():
            for data_tuple in tqdm(self.val_loader, desc="Val"):
                
                inputs, masks = data_tuple[0].to(self.device), data_tuple[1].to(self.device)
                
                outputs = self.net(inputs)
                
                loss = self.loss_func(outputs, masks)

                self.update_metrics(
                    split = "val",
                    batch_size = inputs.size(0),
                    loss = loss.item(),
                    dice = dice_coefficient(outputs.detach(), masks.detach()),
                    iou = iou_score(outputs.detach(), masks.detach()),
                    accuracy = pixel_accuracy(outputs.detach(), masks.detach()))
        
        ret = self.model_utility.save(
            self.metrics[self.configer.model_config.get("checkpoints_metric")]["val"].avg,
            self.net,
            self.optimizer,
            self.epoch + 1,
            self.scheduler)

        if ret < 0:
            return -1
        return ret

    def train(self):
        
        self.net.freeze_encoder()
        
        for n in range(self.configer['epochs']):
            print("Starting epoch {} of {}.".format(self.epoch + 1, self.configer['epochs'] + self.epoch_init))
            
            if n == self.epoch_init + self.encoder_start_epoch:
                self.net.unfreeze_encoder()
                
                print(f"Encoder training started (finetune next {self.configer.model_config['encoder_tune_epochs']} epochs).")
            
            self.__train()
            ret = self.__val()
            
            if self.scheduler is not None:
                self.scheduler.step(self.metrics[self.configer.model_config.get("checkpoints_metric")]['val'].avg)
                print('lr_0:', self.optimizer.param_groups[0]["lr"])
                print('lr_1:', self.optimizer.param_groups[1]["lr"])

            self.log_epoch_history(['train', 'val'])
            self.print_metrics(['train', 'val'])
            self.reset_metrics()

            if ret < 0:
                print("Got no improvement for {} subsequent epochs. Finished epoch {}, than stopped."
                      .format(self.configer.model_config.get("early_stop_number"), self.epoch_init + n+1))
                break
            
            self.epoch += 1
        
        return self.train_history, \
            len(self.train_loader.dataset), \
            len(self.val_loader.dataset), \
            self.model_size