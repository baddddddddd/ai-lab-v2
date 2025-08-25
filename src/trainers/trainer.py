import math
from typing import Optional, Callable, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, IterableDataset, DataLoader

from ..utils.schedulers import LinearWarmupLR
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
        scheduler: optim.lr_scheduler._LRScheduler | None = None,
    ):
        self.model = model
        self.args = args
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler

        if self.args.device is not None:
            self.device = torch.device(self.args.device)
        else:
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        self.model.to(self.device)

    def get_optimizer(self):
        if self.optimizer is None:
            self.optimizer = self._create_optimizer()

        return self.optimizer

    def get_scheduler(self):
        if self.scheduler is None:
            self.scheduler = self._create_scheduler()

        return self.scheduler

    def get_train_dataloader(self):
        if self.train_dataloader is None:
            self.train_dataloader = self._create_train_dataloader()

        return self.train_dataloader

    def get_eval_dataloader(self):
        if self.eval_dataloader is None:
            self.eval_dataloader = self._create_eval_dataloader()

        return self.eval_dataloader

    def _create_optimizer(self):
        return optim.AdamW(
            params=self.model.parameters(),
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
            weight_decay=self.args.weight_decay,
        )

    def _create_scheduler(self):
        if self.args.warmup_steps is None:
            self.args.warmup_steps = math.ceil(
                self.args.max_steps * self.args.warmup_ratio
            )

        return LinearWarmupLR(
            optimizer=self.optimizer,
            warmup_steps=self.args.warmup_steps,
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

    def _prepare_inputs(self, inputs: dict) -> dict:
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        return inputs

    def _compute_loss(self, inputs: dict) -> torch.FloatTensor:
        output = self.model.forward(**inputs)
        loss = output.loss
        return loss

    def _training_step(self, inputs: dict):
        self.model.train()
        inputs = self._prepare_inputs(inputs)

        loss = self._compute_loss(inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        return loss.detach()

    def _optimizer_step(self):
        if self.args.max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.args.max_grad_norm
            )

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        self.scheduler.step()

    def train(self, resume_from_checkpoint: bool = False):
        if self.train_dataloader is None:
            self.train_dataloader = self._create_train_dataloader()

        num_batch_per_epoch = len(self.train_dataloader)
        num_steps_per_epoch = math.ceil(
            num_batch_per_epoch / self.args.gradient_accumulation_steps
        )

        if self.args.max_steps is None:
            self.args.max_steps = math.ceil(
                self.args.num_train_epochs * num_steps_per_epoch
            )

        if self.optimizer is None:
            self.optimizer = self._create_optimizer()

        if self.scheduler is None:
            self.scheduler = self._create_scheduler()

        optimizer_step_count = 0  # TODO: Update this when resuming from checkpoint
        accumulated_steps = 0
        while optimizer_step_count < self.args.max_steps:
            for batch_idx, inputs in enumerate(self.train_dataloader):
                loss = self._training_step(inputs)
                accumulated_steps += 1

                if accumulated_steps == self.args.gradient_accumulation_steps:
                    self._optimizer_step()
                    optimizer_step_count += 1
                    accumulated_steps = 0

                    if optimizer_step_count >= self.args.max_steps:
                        break

            # TODO: Make sure that this always matches what's inside the loop
            if accumulated_steps > 0 and optimizer_step_count < self.args.max_steps:
                self._optimizer_step()
                optimizer_step_count += 1
                accumulated_steps = 0
