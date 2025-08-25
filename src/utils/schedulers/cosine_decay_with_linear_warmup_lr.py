import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CosineDecayWithLinearWarmupLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay_steps = total_steps - warmup_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            factor = self.last_epoch / self.warmup_steps
            return [base_lr * factor for base_lr in self.base_lrs]

        elif self.last_epoch < self.total_steps:
            progress = (self.last_epoch - self.warmup_steps) / self.decay_steps
            factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * factor
                for base_lr in self.base_lrs
            ]

        else:
            return [self.min_lr for _ in self.base_lrs]
