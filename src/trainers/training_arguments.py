import os
from pathlib import Path


class TrainingArguments:
    def __init__(
        self,
        output_dir: os.PathLike,
        train_batch_size: int = 1,
        eval_batch_size: int = 1,
        gradient_accumulation_steps: int = 1,
        num_train_epochs: float = 3.0,
        max_steps: int | None = None,
        learning_rate: float = 5e-05,
        weight_decay: float = 0.0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-08,
        max_grad_norm: float = 1.0,
        warmup_ratio: float = 0.0,
        warmup_steps: int | None = None,
        device: str | None = None,
        dataloader_drop_last: bool = False,
        dataloader_num_workers: int = 0,
        dataloader_prefetch_factor: int | None = None,
        dataloader_pin_memory: bool = True,
        dataloader_persistent_workers: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.max_grad_norm = max_grad_norm
        self.warmup_ratio = warmup_ratio
        self.warmup_steps = warmup_steps

        self.device = device

        self.dataloader_drop_last = dataloader_drop_last
        self.dataloader_num_workers = dataloader_num_workers
        self.dataloader_prefetch_factor = dataloader_prefetch_factor
        self.dataloader_pin_memory = dataloader_pin_memory
        self.dataloader_persistent_workers = dataloader_persistent_workers
