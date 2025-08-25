from typing import Optional, Callable, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, IterableDataset, DataLoader

from .training_arguments import TrainingArguments


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        data_collator: Callable | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | None = None,
        train_dataloader: DataLoader | None = None,
        eval_dataloader: DataLoader | None = None,
        optimizer: optim.Optimizer | None = None,
    ):
        self.model = model
        self.args = args
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer

    def _create_optimizer(self):
        return optim.AdamW(
            params=self.model.parameters(),
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
            weight_decay=self.args.weight_decay,
        )

    def _create_train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.args.train_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            prefetch_factor=self.args.dataloader_prefetch_factor,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
        )

    def _create_eval_dataloader(self):
        return DataLoader(
            dataset=self.eval_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            prefetch_factor=self.args.dataloader_prefetch_factor,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
        )

    def _training_step(self):
        pass

    def train(self, resume_from_checkpoint: bool = False):
        if self.optimizer is None:
            self.optimizer = self._create_optimizer()

        if self.train_dataloader is None:
            self.train_dataloader = self._create_train_dataloader()
