import math
import torch

class WarmupInvRsqrtLR(torch.optim.lr_scheduler._LRScheduler):
    # Планировщик с линейным прогревом и обратным квадратным корнем.
    # На этапе прогрева LR растёт линейно до lr_max, затем убывает как 1/sqrt(step).
    def __init__(self, optimizer, lr_max: float, warmup_steps: int, last_epoch: int = -1):
        """
        Args:
            optimizer: Оптимизатор, к которому привязывается планировщик.
            lr_max: Максимальная скорость обучения (достигается на warmup_steps).
            warmup_steps: Количество шагов, за которое LR достигает lr_max.
            last_epoch: Индекс последнего шага. Нужен для корректного восстановления из чекпоинта.
        """
        self._lr_max = lr_max
        self._warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch=last_epoch)

    def current_rate(self) -> float:
        step = self.last_epoch
        
        # На шаге 0 возвращаем 0, чтобы избежать деления на ноль в decay_factor.
        if step == 0:
            return 0.0
            
        # Линейный прогрев: lr = lr_max * (step / warmup_steps).
        warmup_factor = step / self._warmup_steps
        
        # Обратный квадратный корень: lr = lr_max * sqrt(warmup_steps / step).
        decay_factor = math.sqrt(self._warmup_steps / step)
        
        # Планировщик выбирает меньшее значение, создавая плавный переход в точке warmup_steps.
        return self._lr_max * min(warmup_factor, decay_factor)

    def get_lr(self):
        return [self.current_rate() for _ in self.optimizer.param_groups]
    

class WarmupCosineDecayLR(torch.optim.lr_scheduler._LRScheduler):
    """Планировщик с линейным прогревом и асимптотическим косинусным затуханием.
    
    После warmup LR уменьшается по косинусной кривой, асимптотически приближаясь к eta_min.
    Не требует знания total_steps (в отличие от CosineAnnealingLR).
    """

    def __init__(self, optimizer, lr_max: float, warmup_steps: int, 
                 decay_rate: float = 1.0, eta_min: float = 0.0, last_epoch: int = -1):
        """
        Args:
            optimizer: Оптимизатор.
            lr_max: Пиковая скорость обучения.
            warmup_steps: Количество шагов для линейного прогрева.
            decay_rate: Скорость затухания (больше = быстрее падает).
            eta_min: Минимальная скорость обучения (асимптота).
            last_epoch: Индекс последнего шага (для чекпоинтов).
        """
        self.lr_max = lr_max
        self.warmup_steps = warmup_steps
        self.decay_rate = decay_rate
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def current_rate(self) -> float:
        step = self.last_epoch
        
        # 1. Фаза линейного прогрева.
        if step < self.warmup_steps:
            factor = step / max(1, self.warmup_steps)
            return self.eta_min + (self.lr_max - self.eta_min) * factor
        
        # 2. Фаза косинусного затухания (асимптотическая).
        # Используем arctan для создания асимптотического прогресса от 0 до 1
        cosine_step = step - self.warmup_steps
        
        # progress растет от 0 до 1 асимптотически (как arctan)
        # decay_rate контролирует, как быстро достигается прогресс.
        progress = math.atan(cosine_step * self.decay_rate / self.warmup_steps) / (math.pi / 2)
        
        # Косинусное затухание: от lr_max (progress=0) до eta_min (progress→1)
        return self.eta_min + 0.5 * (self.lr_max - self.eta_min) * (1 + math.cos(math.pi * progress))

    def get_lr(self):
        return [self.current_rate() for _ in self.optimizer.param_groups]

class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    """Планировщик с линейным прогревом и косинусным изменением."""

    def __init__(self, optimizer, lr_max: float, warmup_steps: int, total_steps: int, 
                 eta_min: float = 0.0, last_epoch: int = -1):
        """
        Args:
            optimizer: Оптимизатор.
            lr_max: Пиковая скорость обучения (достигается в конце прогрева).
            warmup_steps: Количество шагов для линейного прогрева.
            total_steps: Общее количество шагов обучения (прогрев + косинус).
            eta_min: Минимальная скорость обучения (в конце обучения).
            last_epoch: Индекс последнего шага (для чекпоинтов).
        """
        self.lr_max = lr_max
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def current_rate(self) -> float:
        step = self.last_epoch
        
        # 1. Фаза линейного прогрева (от eta_min до lr_max)
        if step < self.warmup_steps:
            factor = step / max(1, self.warmup_steps)
            return self.eta_min + (self.lr_max - self.eta_min) * factor
            
        # 2. Фаза косинусного затухания (от lr_max до eta_min)
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, progress)  # Ограничиваем на случай, если step > total_steps
        
        return self.eta_min + 0.5 * (self.lr_max - self.eta_min) * (1.0 + math.cos(math.pi * progress))

    def get_lr(self):
        return [self.current_rate() for _ in self.optimizer.param_groups]


class WarmupCosineAnnealingWarmRestarts(torch.optim.lr_scheduler._LRScheduler):
    """Планировщик с линейным прогревом, косинусным изменением и рестартами.
    
    После warmup начинается первый косинусный цикл длиной T_0.
    Каждый следующий цикл в T_mult раз длиннее предыдущего.
    """

    def __init__(self, optimizer, lr_max: float, warmup_steps: int, 
                 T_0: int, T_mult: int = 1, eta_min: float = 0.0, last_epoch: int = -1):
        """
        Args:
            optimizer: Оптимизатор.
            lr_max: Пиковая скорость обучения.
            warmup_steps: Количество шагов для линейного прогрева.
            T_0: Длина первого косинусного цикла (после warmup).
            T_mult: Множитель длины цикла (1 = постоянная длина, 2 = удвоение каждый раз).
            eta_min: Минимальная скорость обучения.
            last_epoch: Индекс последнего шага (для чекпоинтов).
        """
        self.lr_max = lr_max
        self.warmup_steps = warmup_steps
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        
        # Счётчик шагов внутри текущего косинусного цикла.
        self.T_cur = 0
        self.T_i = T_0  # Длина текущего цикла.
        
        super().__init__(optimizer, last_epoch)

    def current_rate(self) -> float:
        step = self.last_epoch
        
        # 1. Фаза линейного прогрева.
        if step < self.warmup_steps:
            factor = step / max(1, self.warmup_steps)
            return self.eta_min + (self.lr_max - self.eta_min) * factor
        
        # 2. Фаза косинусных рестартов.
        # Вычисляем, в каком цикле мы находимся и какой шаг внутри цикла.
        cosine_step = step - self.warmup_steps
        
        # Определяем T_cur (шаг внутри текущего цикла) и T_i (длину текущего цикла).
        T_cur = cosine_step
        T_i = self.T_0
        
        # Находим текущий цикл.
        while T_cur >= T_i:
            T_cur -= T_i
            T_i *= self.T_mult
        
        # Косинусная формула (стандартная из PyTorch).
        return self.eta_min + 0.5 * (self.lr_max - self.eta_min) * (1 + math.cos(math.pi * T_cur / T_i))

    def get_lr(self):
        return [self.current_rate() for _ in self.optimizer.param_groups]
    
