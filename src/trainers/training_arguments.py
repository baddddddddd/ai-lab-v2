import os
from pathlib import Path


class TrainingArguments:
    def __init__(
        self,
        output_dir: os.PathLike,
        train_batch_size: int = 1,
        eval_batch_size: int = 1,
        learning_rate: float = 5e-05,
        weight_decay: float = 0.0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-08,
        dataloader_drop_last: bool = False,
        dataloader_num_workers: int = 0,
        dataloader_prefetch_factor: int | None = None,
        dataloader_pin_memory: bool = True,
        dataloader_persistent_workers: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon

        self.dataloader_drop_last = dataloader_drop_last
        self.dataloader_num_workers = dataloader_num_workers
        self.dataloader_prefetch_factor = dataloader_prefetch_factor
        self.dataloader_pin_memory = dataloader_pin_memory
        self.dataloader_persistent_workers = dataloader_persistent_workers
