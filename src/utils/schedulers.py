import math
import torch

class WarmupInvRsqrtLR(torch.optim.lr_scheduler._LRScheduler):
    # Scheduler with linear warmup and inverse square root decay.
    # During the warmup phase, LR grows linearly to lr_max, then decays as 1/sqrt(step).
    def __init__(self,
        optimizer,
        lr_max: float,
        warmup_steps: int,
        last_epoch: int = -1
        ):
        """
        Args:
            optimizer: The optimizer to bind the scheduler to.
            lr_max: Maximum learning rate (reached at warmup_steps).
            warmup_steps: Number of steps over which the LR reaches lr_max.
            last_epoch: The index of the last step. Needed for correct resuming from a checkpoint.
        """
        self._lr_max = lr_max
        self._warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch=last_epoch)

    def current_rate(self) -> float:
        step = self.last_epoch
        
        # Avoid division by zero in decay_factor.
        if step == 0:
            return 0.0
            
        # Linear warmup.
        warmup_factor = step / self._warmup_steps
        
        # Inv sqrt decay after warmup.
        decay_factor = math.sqrt(self._warmup_steps / step)
        
        return self._lr_max * min(warmup_factor, decay_factor)

    def get_lr(self):
        return [self.current_rate() for _ in self.optimizer.param_groups]
    

class WarmupCosineDecayLR(torch.optim.lr_scheduler._LRScheduler):
    """Scheduler with linear warmup and asymptotic cosine decay.
    
    After warmup, the LR decreases along a cosine curve, asymptotically approaching eta_min.
    Does not require knowing total_steps (unlike CosineAnnealingLR).
    """

    def __init__(self,
        optimizer,
        lr_max: float,
        warmup_steps: int, 
        decay_rate: float = 1.0,
        eta_min: float = 0.0,
        last_epoch: int = -1
        ):
        """
        Args:
            optimizer: The optimizer.
            lr_max: Peak learning rate.
            warmup_steps: Number of steps for linear warmup.
            decay_rate: Decay rate (higher = faster drop).
            eta_min: Minimum learning rate (asymptote).
            last_epoch: Index of the last step (for checkpoints).
        """
        self.lr_max = lr_max
        self.warmup_steps = warmup_steps
        self.decay_rate = decay_rate
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def current_rate(self) -> float:
        step = self.last_epoch
        
        # 1. Linear warmup phase.
        if step < self.warmup_steps:
            factor = step / max(1, self.warmup_steps)
            return self.eta_min + (self.lr_max - self.eta_min) * factor
        
        # 2. Cosine decay phase (asymptotic).
        # Use arctan to create an asymptotic progress from 0 to 1.
        cosine_step = step - self.warmup_steps
        
        # progress grows from 0 to 1 asymptotically (like arctan).
        # decay_rate controls how fast the progress is reached.
        progress = math.atan(cosine_step * self.decay_rate / self.warmup_steps) / (math.pi / 2)
        
        # Cosine decay: from lr_max (progress=0) to eta_min (progress→1).
        return self.eta_min + (self.lr_max - self.eta_min) * (1 + math.cos(math.pi/2 * (progress + 1)))

    def get_lr(self):
        return [self.current_rate() for _ in self.optimizer.param_groups]


class WarmupCosineAnnealingWarmRestarts(torch.optim.lr_scheduler._LRScheduler):
    """Scheduler with linear warmup, cosine annealing, and warm restarts.
    
    After warmup, the first cosine cycle of length T_0 begins.
    Each subsequent cycle is T_mult times longer than the previous one.
    """

    def __init__(self,
        optimizer,
        restart_flag: bool,
        lr_max: float,
        warmup_steps: int,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0.0,
        last_epoch: int = -1
        ):
        
        """
        Args:
            optimizer: The optimizer.
            lr_max: Peak learning rate.
            warmup_steps: Number of steps for linear warmup.
            T_0: Length of the first cosine cycle (after warmup).
            T_mult: Cycle length multiplier (1 = constant length, 2 = doubling each time).
            eta_min: Minimum learning rate.
            last_epoch: Index of the last step (for checkpoints).
        """
        self.lr_max = lr_max
        self.warmup_steps = warmup_steps
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        
        # Step counter inside the current cosine cycle.
        self.T_cur = 0
        self.T_i = T_0  # Length of the current cycle.
        
        if restart_flag:
            self.mult = 1.0
        else:
            self.mult = 2.0
        
        super().__init__(optimizer, last_epoch)

    def current_rate(self) -> float:
        step = self.last_epoch
        
        # 1. Linear warmup phase.
        if step < self.warmup_steps:
            factor = step / max(1, self.warmup_steps)
            return self.eta_min + (self.lr_max - self.eta_min) * factor
        
        # 2. Cosine restarts phase.
        # Calculate which cycle we are in and the step inside the cycle.
        cosine_step = step - self.warmup_steps
        
        # Determine T_cur (step inside the current cycle) and T_i (length of the current cycle).
        T_cur = cosine_step
        T_i = self.T_0
        
        # Find the current cycle.
        while T_cur >= T_i:
            T_cur -= T_i
            T_i *= self.T_mult
        
        # Cosine formula (standard from PyTorch).
        return self.eta_min + 0.5 * (self.lr_max - self.eta_min) * (1 + math.cos(self.mult * math.pi * T_cur / T_i))

    def get_lr(self):
        return [self.current_rate() for _ in self.optimizer.param_groups]