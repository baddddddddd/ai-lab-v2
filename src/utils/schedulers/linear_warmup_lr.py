from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupLR(_LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup_steps: int, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            factor = self.last_epoch / self.warmup_steps
            return [base_lr * factor for base_lr in self.base_lrs]
        return [base_lr for base_lr in self.base_lrs]
