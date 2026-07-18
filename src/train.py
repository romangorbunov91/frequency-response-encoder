import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # For deterministic cuBLAS
from pathlib import Path
import json

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
torch.use_deterministic_algorithms(True)

# Import Utils.
from src.utils.metrics import AverageMeter, CombinedLoss, dice_coefficient, iou_score, pixel_accuracy
from src.utils.debug_functions import visualize_predictions, print_terminal_graph
from src.utils.logging_functions import build_output_dict
from src.utils.schedulers import WarmupInvRsqrtLR, WarmupCosineDecayLR, WarmupCosineAnnealingWarmRestarts

# Import Datasets.
from torch.utils.data import DataLoader
from src.dataloaders.ZerosPolesDataset import TransformsConfig, ZerosPolesDataset, ConversionTransforms, GeneralTransforms

# Import Model.
from src.models.model_utilizer import load_net, update_optimizer_simple, ModelUtilizer
from src.models.base_model import base_model

# Setting seeds.
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    #random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(worker_seed)

class MetricTracker:
    def __init__(self, metric_names, phases):
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
    
    def print_metrics(self, phases):
        phase_keys = [phase.lower() for phase in phases]
        
        if len(self.train_history['epoch']) > 0:
            prefix = f"Epoch {self.train_history['epoch'][-1]:2d} | "
        else:
            prefix = ''
        
        if 'train' in phase_keys:
            print_str = ", ".join(f"{name}: {self.train_history[f'train_{name}'][-1]:.4f}" 
                                for name in list(self.metrics.keys()))
            print(f"{prefix}TRAIN {print_str}")
        
        if 'val' in phase_keys:
            print_str = ", ".join(f"{name}: {self.train_history[f'val_{name}'][-1]:.4f}" 
                                for name in list(self.metrics.keys()))
            print(f"{' ' * len(prefix)}VAL   {print_str}")

        if 'test' in phase_keys:
            print_str = ", ".join(f"{name}: {self.train_history[f'test_{name}'][-1]:.4f}" 
                                for name in list(self.metrics.keys()))
            print(f"{' ' * len(prefix)}TEST  {print_str}")
            
    def log_epoch_history(self, phases, epoch: int, lr: float):
        
        self.train_history["epoch"].append(epoch)
        self.train_history["lr"].append(lr)
        
        phase_keys = [phase.lower() for phase in phases]
        for name in self.metrics:
            for split in phase_keys:
                key = f"{split}_{name}"
                self.train_history[key].append(self.metrics[name][split].avg)

    def update_metrics(self, split: str, batch_size: int, **metrics: float) -> None:
        """
        Update metrics for a given split.
        
        Args:
            split (str): 'train', 'val', 'test'.
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

class ModelTrainer:

    def __init__(self, configer):
        
        self.configer = configer
        
        #: str: Type of dataset.
        self.dataset_family = self.configer["dataset"]["dataset_family"].lower()
        self.dataset_path = Path(self.configer.general_config["data_dir"]) / \
                            self.configer["dataset"]["dataset_name"].lower()
        
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
        self.optimizer = None
        self.scheduler = None
        self.loss = None
        
        self.mask_threshold = self.configer["training"]["mask_threshold"]
        self.bce_weight = self.configer["training"]["bce_weight"]
        self.dice_weight = self.configer["training"]["dice_weight"]
        self.ds_weights = self.configer["training"]['ds_weights']

        transforms_dict = self.configer["transforms"]
        conversions_dict = self.configer["conversions"]

        # Transforms.
        self.train_transforms = [
            GeneralTransforms(config=TransformsConfig(**transforms_dict)),
            ConversionTransforms(**conversions_dict)]

        self.val_transforms = [
            ConversionTransforms(**conversions_dict)]
        
        self.test_transforms = [
            ConversionTransforms(**conversions_dict)]

        if self.configer["model"]["model_name"] == 'base-model':
            self.model_type = base_model
        else:
            raise NotImplementedError(f"Model '{self.configer['model_name']}' is not supported.")
        
        self.metric_tracker = MetricTracker(
            metric_names=['loss', 'dice', 'iou', 'accuracy'],
            phases=['train', 'val', 'test']
        )
        
        if self.configer.general_config['debug_lr_log']:
            self.lr_debug_history = {key: [] for key in ["epoch", "lr"]}


    def init_model(self):
        """Initialize model and other data for procedure"""
        
        # Setting model and loss.
        mdl_input_size = self.configer["model"]['input_size']
        mdl_output_size = self.configer["model"]['output_size']

        self.net = self.model_type(
            in_channels = mdl_input_size[0],
            out_channels = mdl_output_size[0],
            features = self.configer["model"]['feature_list']
            )
        
        self.loss_func = CombinedLoss(
            bce_weight=self.bce_weight,
            dice_weight=self.dice_weight,
            ds_weights=self.ds_weights
            ).to(self.device)

        # Initializing training.
        self.net, self.epoch_init, optim_dict, sched_dict = load_net(
            net = self.net,
            checkpoints_file = self.configer['resume'],
            device = self.device
            )
        self.epoch = self.epoch_init
        
        # Setting optimizer.
        self.optimizer = update_optimizer_simple(
            net = self.net,
            optim = self.configer["training"]['solver_type'],
            lr = self.configer["training"]['base_lr'],
            weight_decay = self.configer["training"]['weight_decay']
            )
        
        if optim_dict is None:
            print(f"Starting training {self.configer['model']['model_name']} from scratch using {self.configer['training']['solver_type']}.")
        else:
            self.optimizer.load_state_dict(optim_dict)
            print(f"Resuming training {self.configer['model']['model_name']} from epoch {self.epoch} using {self.configer['training']['solver_type']}.")
        
        # Setting scheduler.
        if self.configer["scheduler"]['scheduler_type'] is not None:
            if self.configer["scheduler"]['scheduler_type'] == "CosineAnnealingLR":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.configer["scheduler"]['scheduler_T_max'],
                    eta_min=self.configer["scheduler"]['scheduler_eta_min']
                    )
            elif self.configer["scheduler"]['scheduler_type'] == "CosineAnnealingWarmRestarts":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer, 
                    T_0 = self.configer["scheduler"]['scheduler_T_0'],
                    T_mult = self.configer["scheduler"]['scheduler_T_mult'],
                    eta_min = self.configer["scheduler"]['scheduler_eta_min']
                    )
            elif self.configer["scheduler"]['scheduler_type'] == "WarmupCosineAnnealing":
                self.scheduler = WarmupCosineAnnealingWarmRestarts(
                    self.optimizer,
                    lr_max=self.configer["training"]['base_lr'],
                    warmup_steps=self.configer['scheduler_warmup_steps'],
                    T_0=self.configer["scheduler"]['scheduler_T_0'],
                    T_mult=self.configer["scheduler"]['scheduler_T_mult'],
                    eta_min = self.configer["scheduler"]['scheduler_eta_min'],
                    restart_flag=False
                    )
            elif self.configer["scheduler"]['scheduler_type'] == "WarmupCosineAnnealingWarmRestarts":
                self.scheduler = WarmupCosineAnnealingWarmRestarts(
                    self.optimizer,
                    lr_max=self.configer["training"]['base_lr'],
                    warmup_steps=self.configer["scheduler"]['scheduler_warmup_steps'],
                    T_0=self.configer["scheduler"]['scheduler_T_0'],
                    T_mult=self.configer["scheduler"]['scheduler_T_mult'],
                    eta_min = self.configer["scheduler"]['scheduler_eta_min'],
                    restart_flag=True
                    )
            elif self.configer["scheduler"]['scheduler_type'] == "WarmupInvRsqrtLR":
                self.scheduler = WarmupInvRsqrtLR(
                    self.optimizer,
                    lr_max=self.configer["training"]['base_lr'],
                    warmup_steps=self.configer["scheduler"]['scheduler_warmup_steps'],
                    eta_min = self.configer["scheduler"]['scheduler_eta_min']
                    )
            elif self.configer["scheduler"]['scheduler_type'] == "WarmupCosineDecayLR":
                self.scheduler = WarmupCosineDecayLR(
                    self.optimizer,
                    lr_max=self.configer["training"]['base_lr'],
                    warmup_steps=self.configer["scheduler"]['scheduler_warmup_steps'],
                    decay_rate=self.configer["scheduler"]['scheduler_decay_rate'],
                    eta_min=self.configer["scheduler"]['scheduler_eta_min']
                    )
            else:
                raise NotImplementedError(f"Scheduler not supported: {self.configer['scheduler']['scheduler_type']}")

            if sched_dict is not None:
                self.scheduler.load_state_dict(sched_dict)
            print(f"Scheduler ON: {self.configer['scheduler']['scheduler_type']}")
        else:
            print(f"Scheduler OFF")
        
        self.model_size = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print(f"Model parameters: {self.model_size}")

        # Setting Dataloaders.
        shuffle_generator = torch.Generator()
        shuffle_generator.manual_seed(self.configer.general_config['seed'])
        
        if self.dataset_family == "zeros-poles-dataset":
            self.train_loader = DataLoader(
                ZerosPolesDataset(
                    dataset_dir=self.dataset_path,
                    split='train',
                    mask_halfwindow=self.configer["training"]["mask_halfwindow"],
                    transforms=self.train_transforms
                    ), 
                batch_size=self.configer["training"]["batch_size"],
                shuffle=True,
                generator=shuffle_generator,
                num_workers=self.configer.general_config["workers"],
                worker_init_fn=worker_init_fn,
                pin_memory=True)

            self.val_loader = DataLoader(
                ZerosPolesDataset(
                    dataset_dir=self.dataset_path,
                    split='val',
                    mask_halfwindow=self.configer["training"]["mask_halfwindow"],
                    transforms=self.val_transforms
                    ), 
                batch_size=self.configer["training"]["batch_size"],
                shuffle=False,
                num_workers=self.configer.general_config["workers"],
                pin_memory=True)
            
            self.test_loader = DataLoader(
                ZerosPolesDataset(
                    dataset_dir=self.dataset_path,
                    split='test',
                    mask_halfwindow=self.configer["training"]["mask_halfwindow"],
                    transforms=self.test_transforms
                    ), 
                batch_size=self.configer["training"]["batch_size"],
                shuffle=False,
                num_workers=self.configer.general_config["workers"],
                pin_memory=True)
            
        else:
            raise NotImplementedError(f"Dataset not supported: {self.dataset_family}")
        
        print(f"TRAIN size: {len(self.train_loader.dataset)}")
        print(f"VAL   size: {len(self.val_loader.dataset)}")
        print(f"TEST  size: {len(self.test_loader.dataset)}")
              
    def __train(self):
        """Train function for every epoch."""
        self.net.train()
        
        for data_tuple in tqdm(self.train_loader, desc="Train"):

            inputs, masks = data_tuple[0].to(self.device), data_tuple[1].to(self.device)
            
            logits = self.net(inputs)

            self.optimizer.zero_grad()
            loss = self.loss_func(
                logits=logits, 
                targets=masks
                )
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            if (self.scheduler is not None) and (self.configer["scheduler"]['scheduler_mode'] == 'batch'):
                self.lr_list.append(self.optimizer.param_groups[0]["lr"])
                self.scheduler.step()

            if isinstance(logits, tuple):
                logits, _ = logits

            self.metric_tracker.update_metrics(
                split = "train",
                batch_size = inputs.size(0),
                loss = loss.item(),
                dice = dice_coefficient(
                    logits=logits.detach(),
                    targets=masks.detach(),
                    threshold=self.mask_threshold
                    ),
                iou = iou_score(
                    logits=logits.detach(),
                    targets=masks.detach(),
                    threshold=self.mask_threshold
                    ),
                accuracy = pixel_accuracy(
                    logits=logits.detach(),
                    targets=masks.detach(),
                    threshold=self.mask_threshold
                    )
                )
        
    def __val(self):
        """Validation function."""
        self.net.eval()

        with torch.no_grad():
            for data_tuple in tqdm(self.val_loader, desc="Val  "):
                
                inputs, masks = data_tuple[0].to(self.device), data_tuple[1].to(self.device)
                
                logits = self.net(inputs)
                
                loss = self.loss_func(
                    logits=logits, 
                    targets=masks
                    )

                if isinstance(logits, tuple):
                    logits, _ = logits

                self.metric_tracker.update_metrics(
                    split = "val",
                    batch_size = inputs.size(0),
                    loss = loss.item(),
                    dice = dice_coefficient(
                        logits=logits.detach(),
                        targets=masks.detach(),
                        threshold=self.mask_threshold
                        ),
                    iou = iou_score(
                        logits=logits.detach(),
                        targets=masks.detach(),
                        threshold=self.mask_threshold
                        ),
                    accuracy = pixel_accuracy(
                        logits=logits.detach(),
                        targets=masks.detach(),
                        threshold=self.mask_threshold
                        )
                )
        
        ret = self.model_utility.save(
            self.metric_tracker.metrics[self.configer["checkpoints"]["checkpoints_metric"]]["val"].avg,
            self.net,
            self.optimizer,
            self.epoch + 1,
            self.scheduler)

        if ret < 0:
            return -1
        return ret

    def __test(self):
        """Test function."""
        self.net.eval()

        with torch.no_grad():
            for data_tuple in tqdm(self.test_loader, desc="Test "):
                
                inputs, masks = data_tuple[0].to(self.device), data_tuple[1].to(self.device)
                
                logits = self.net(inputs)
                
                loss = self.loss_func(
                    logits=logits, 
                    targets=masks
                    )
                
                if isinstance(logits, tuple):
                    logits, _ = logits

                self.metric_tracker.update_metrics(
                    split = "test",
                    batch_size = inputs.size(0),
                    loss = loss.item(),
                    dice = dice_coefficient(
                        logits=logits.detach(),
                        targets=masks.detach(),
                        threshold=self.mask_threshold
                        ),
                    iou = iou_score(
                        logits=logits.detach(),
                        targets=masks.detach(),
                        threshold=self.mask_threshold
                        ),
                    accuracy = pixel_accuracy(
                        logits=logits.detach(),
                        targets=masks.detach(),
                        threshold=self.mask_threshold
                        )
                    )
        
        # START debug section.
        if self.configer.general_config['debug_num_samples'] > 0:
            num_samples = self.configer.general_config['debug_num_samples']
            batch_size = logits.shape[0]
            random_indices = torch.randperm(batch_size)[:num_samples]
            
            rand_int = random_indices[0]
            print('Prediction:', ((torch.sigmoid(logits[rand_int]) > self.mask_threshold).float()).sum(dim=1).detach())
            print('Ground:    ', masks[rand_int].sum(dim=1).detach())
            
            logits = logits[random_indices]
            masks = masks[random_indices]
            
            save_dir = Path(self.configer.general_config['score_dir']) / f"{self.configer['model']['model_name']}_{self.configer.run_id}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                
            visualize_predictions(
                logits_all=logits.detach(),
                masks_all=masks.detach(),
                dice_func=dice_coefficient,
                iou_func=iou_score,
                save_path=save_dir / f"{self.configer['model']['model_name']}_{self.configer.run_id}_{self.epoch}.pdf",
                threshold=self.mask_threshold
                )
        # END debug section.
            
    def train(self) -> None:
        for n in range(self.configer['training']['epochs']):
            print("Starting epoch {} of {}.".format(self.epoch + 1, self.configer['training']['epochs'] + self.epoch_init))
            print('Learning rate:', self.optimizer.param_groups[0]["lr"])
            
            # Reset learning rate list for this epoch.
            self.lr_list = []
            
            self.__train()
            val_return = self.__val()
            self.__test()

            self.metric_tracker.log_epoch_history(
                phases=['train', 'val', 'test'], 
                epoch=self.epoch + 1, 
                lr=self.optimizer.param_groups[0]["lr"]
                )
            self.metric_tracker.print_metrics(
                phases=['train', 'val', 'test']
                )
            self.metric_tracker.reset_metrics()
            
            if (self.scheduler is not None) and (self.configer["scheduler"]['scheduler_mode'] == 'epoch'):
                self.lr_list = [self.optimizer.param_groups[0]["lr"]]
                self.scheduler.step()
            
            if self.configer.general_config['debug_terminal_graph_lines'] > 0:
                print_terminal_graph(
                    data=self.lr_list,
                    title=f"{self.configer['scheduler']['scheduler_type']} at epoch {self.epoch + 1}",
                    num_lines=self.configer.general_config['debug_terminal_graph_lines']
                    )

            if val_return < 0:
                print("Got no improvement for {} subsequent epochs. Finished epoch {}, than stopped."
                      .format(self.configer['training']['early_stop_number'], self.epoch_init + n+1))
                break
            
            output_dict = build_output_dict(
                configer=self.configer,
                train_history=self.metric_tracker.train_history,
                run_id=self.configer.run_id,
                train_size=len(self.train_loader.dataset),
                val_size=len(self.val_loader.dataset),
                test_size=len(self.test_loader.dataset),
                model_param_count=self.model_size)
            
            with open(Path(self.configer.general_config["logs_dir"]) / (self.configer.output_file_name + '.json'), "w") as f:
                json.dump(output_dict, f, indent=4)
            
            if self.configer.general_config['debug_lr_log']:
                
                self.lr_debug_history["epoch"].append(self.epoch + 1)
                self.lr_debug_history["lr"].append(self.lr_list)
                
                with open(Path(self.configer.general_config["logs_dir"]) / (self.configer.output_file_name + '_lr.json'), "w") as f:
                    json.dump(self.lr_debug_history, f, indent=4)
                    
            self.epoch += 1