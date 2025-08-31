import itertools
import json
import math
import os
import pathlib
import random
import re
from typing import Optional, Callable, Any

from rich.console import Console
from rich.live import Live
from rich.progress import (
    Progress,
    TaskID,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, IterableDataset, DataLoader

from ..models import BaseModel
from ..utils.schedulers import LinearWarmupLR
from .training_arguments import TrainingArguments


class Trainer:
    TRAINER_STATE_FILENAME = "trainer_state.json"
    TRAINING_ARGS_FILENAME = "training_args.bin"
    OPTIMIZER_FILENAME = "optimizer.pt"
    SCHEDULER_FILENAME = "scheduler.pt"
    RNG_STATE_FILENAME = "rng_state.pth"
    SCALER_FILENAME = "scaler.pt"

    def __init__(
        self,
        model: BaseModel | None = None,
        args: TrainingArguments | None = None,
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

        self.scaler = None
        if self.args.fp16:
            if not torch.cuda.is_available():
                raise ValueError("FP16 training requires CUDA")
            self.scaler = torch.cuda.amp.GradScaler()

        # logging
        self.current_step = 0
        self.current_epoch = 0
        self.start_batch_idx = 0
        self._accumulated_loss = 0.0
        self._loss_steps = 0
        self._accumulated_grad_norm = 0.0
        self._grad_norm_steps = 0

        self.console = Console()

        self.progress = Progress(
            TextColumn("[bold blue]Training", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            TextColumn("[bold green]{task.completed}/{task.total} steps"),
            "•",
            TimeElapsedColumn(),
            "•",
            TimeRemainingColumn(),
            console=self.console,
            expand=True,
        )

        self.epoch_progress = Progress(
            TextColumn("[bold magenta]Epoch", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            TextColumn("[bold yellow]{task.completed}/{task.total} batches"),
            console=self.console,
            expand=True,
        )

        self.metrics_data = []
        self.max_table_rows = 20

        self.layout = Layout()
        self.layout.split_column(
            Layout(
                Panel(self.progress, title="Overall Progress", border_style="blue"),
                size=3,
            ),
            Layout(
                Panel(
                    self.epoch_progress, title="Current Epoch", border_style="magenta"
                ),
                size=3,
            ),
            Layout(
                Panel(
                    self._create_metrics_table(), title="Metrics", border_style="green"
                )
            ),
        )

        self.live = Live(self.layout, console=self.console, auto_refresh=False)
        self.progress_task: Optional[TaskID] = None
        self.epoch_task: Optional[TaskID] = None

    def _create_metrics_table(self) -> Table:
        table = Table(show_header=True, title="Training Metrics")
        table.add_column("Epoch", justify="right", style="blue")
        table.add_column("Step", justify="right", style="cyan")
        table.add_column("Training Loss", justify="right", style="magenta")
        table.add_column("Learning Rate", justify="right", style="yellow")
        table.add_column("Gradient Norm", justify="right", style="green")

        recent_data = (
            self.metrics_data[-self.max_table_rows :] if self.metrics_data else []
        )
        for epoch, step, loss, lr, grad_norm in recent_data:
            table.add_row(
                f"{epoch:.2f}",
                str(step),
                f"{loss:.6f}",
                f"{lr:.4e}",
                f"{grad_norm:.6f}",
            )

        return table

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
        if self.args.fp16:
            with torch.cuda.amp.autocast():
                output = self.model.forward(**inputs)
                loss = output.loss
        else:
            output = self.model.forward(**inputs)
            loss = output.loss
        return loss

    def _training_step(self, inputs: dict):
        self.model.train()
        inputs = self._prepare_inputs(inputs)

        loss = self._compute_loss(inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.detach()

    def _optimizer_step(self):
        if self.args.fp16:
            self.scaler.unscale_(self.optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.args.max_grad_norm
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.args.max_grad_norm
            )
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

        return grad_norm.item()

    def _update_loss_tracking(self, loss: torch.Tensor):
        self._accumulated_loss += loss.item()
        self._loss_steps += 1

    def _update_grad_norm_tracking(self, grad_norm: float):
        self._accumulated_grad_norm += grad_norm
        self._grad_norm_steps += 1

    def _reset_loss_tracking(self):
        self._accumulated_loss = 0.0
        self._loss_steps = 0

    def _reset_grad_norm_tracking(self):
        self._accumulated_grad_norm = 0.0
        self._grad_norm_steps = 0

    def _refresh_log_metrics(self):
        self.layout.split_column(
            Layout(
                Panel(self.progress, title="Overall Progress", border_style="blue"),
                size=3,
            ),
            Layout(
                Panel(
                    self.epoch_progress,
                    title=f"Epoch {int(self.current_epoch + 1)}",
                    border_style="magenta",
                ),
                size=3,
            ),
            Layout(
                Panel(
                    self._create_metrics_table(),
                    title="Metrics",
                    border_style="green",
                )
            ),
        )

    def _update_progress_and_log(
        self,
        batch_idx: int,
        total_batches: int,
    ):
        self.progress.update(self.progress_task, advance=1)
        self.epoch_progress.update(self.epoch_task, completed=batch_idx + 1)

        should_log_metrics = self.current_step % self.args.logging_steps == 0

        if should_log_metrics:
            avg_loss = (
                self._accumulated_loss
                / self._loss_steps
                * self.args.gradient_accumulation_steps
            )
            avg_grad_norm = self._accumulated_grad_norm / self._grad_norm_steps
            cur_lr = self.scheduler.get_last_lr()[0]

            self.metrics_data.append(
                (self.current_epoch, self.current_step, avg_loss, cur_lr, avg_grad_norm)
            )

            self._refresh_log_metrics()
            self._reset_loss_tracking()
            self._reset_grad_norm_tracking()

        self.live.refresh()

    def _save_model(self, save_folder: pathlib.Path):
        self.model.save_pretrained(
            save_directory=save_folder,
            overwrite=self.args.overwrite_output_dir,
        )

    def _load_model(self, checkpoint_folder: pathlib.Path):
        self.model = self.model.from_pretrained(
            checkpoint_folder, device_map=self.device
        )

    def _save_optimizer(self, save_folder: pathlib.Path):
        optimizer_file = save_folder / Trainer.OPTIMIZER_FILENAME
        optimizer_state = self.optimizer.state_dict()
        torch.save(optimizer_state, optimizer_file)

    def _load_optimizer(self, checkpoint_folder: pathlib.Path):
        # NOTE: This  might be a problem in the future when we start using custom optimizers
        self.optimizer = self._create_optimizer()

        optimizer_file = checkpoint_folder / Trainer.OPTIMIZER_FILENAME
        if optimizer_file.exists():
            optimizer_state = torch.load(optimizer_file, map_location=self.device)
            self.optimizer.load_state_dict(optimizer_state)

    def _save_scheduler(self, save_folder: pathlib.Path):
        scheduler_file = save_folder / Trainer.SCHEDULER_FILENAME
        scheduler_state = self.scheduler.state_dict()
        torch.save(scheduler_state, scheduler_file)

    def _load_scheduler(self, checkpoint_folder: pathlib.Path):
        if self.scheduler is None:
            self.scheduler = self._create_scheduler()

        scheduler_file = checkpoint_folder / Trainer.SCHEDULER_FILENAME
        scheduler_state = torch.load(scheduler_file, weights_only=True)
        self.scheduler.load_state_dict(scheduler_state)

        # hacky way to sync the scheduler and optimizer lr right away, dont hate me
        self.scheduler.last_epoch -= 1
        self.scheduler.step()

    def _save_scaler(self, save_folder: pathlib.Path):
        if self.args.fp16 and self.scaler is not None:
            scaler_file = save_folder / Trainer.SCALER_FILENAME
            scaler_state = self.scaler.state_dict()
            torch.save(scaler_state, scaler_file)

    def _load_scaler(self, checkpoint_folder: pathlib.Path):
        if self.args.fp16:
            if self.scaler is None:
                self.scaler = torch.cuda.amp.GradScaler()

            scaler_file = checkpoint_folder / Trainer.SCALER_FILENAME
            if scaler_file.exists():
                scaler_state = torch.load(scaler_file, weights_only=True)
                self.scaler.load_state_dict(scaler_state)

    def _save_rng_state(self, save_folder: pathlib.Path):
        rng_state_file = save_folder / Trainer.RNG_STATE_FILENAME
        rng_state = {
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "dataloader_state": (
                self.train_dataloader.sampler.state_dict()
                if hasattr(self.train_dataloader.sampler, "state_dict")
                else None
            ),
        }

        torch.save(rng_state, rng_state_file)

    def _load_rng_state(self, checkpoint_folder: pathlib.Path):
        if self.train_dataloader is None:
            self.train_dataloader = self._create_train_dataloader()

        rng_state_file = checkpoint_folder / Trainer.RNG_STATE_FILENAME
        rng_state = torch.load(rng_state_file, weights_only=False)
        torch.set_rng_state(rng_state["torch_rng_state"])
        torch.cuda.set_rng_state_all(rng_state["cuda_rng_state"])
        np.random.set_state(rng_state["numpy_rng_state"])
        random.setstate(rng_state["python_rng_state"])
        if rng_state["dataloader_state"] and hasattr(
            self.train_dataloader.sampler, "load_state_dict"
        ):
            self.train_dataloader.sampler.load_state_dict(rng_state["dataloader_state"])

    def _save_training_args(self, save_folder: pathlib.Path):
        training_args_file = save_folder / Trainer.TRAINING_ARGS_FILENAME
        torch.save(self.args, training_args_file)

    def _save_trainer_state(self, save_folder: pathlib.Path):
        trainer_state_file = save_folder / Trainer.TRAINER_STATE_FILENAME

        log_history = []
        for epoch, step, loss, lr, grad_norm in self.metrics_data:
            log_history.append(
                {
                    "epoch": epoch,
                    "step": step,
                    "train_loss": loss,
                    "learning_rate": lr,
                    "grad_norm": grad_norm,
                }
            )

        trainer_state = {
            "global_step": self.current_step,
            "epoch": self.current_epoch,
            "start_batch_idx": self.start_batch_idx,
            "max_steps": self.args.max_steps,
            "num_train_epochs": self.args.num_train_epochs,
            "log_history": log_history,
            # "total_flos": 0,
            # "best_metric": None,
            # "best_model_checkpoint": None,
            # "trial_name": None,
            # "trial_params": None,
            # "is_local_process_zero": True,
            # "is_world_process_zero": True,
        }

        with open(trainer_state_file, "w") as f:
            json.dump(trainer_state, f, indent=2)

    def _load_trainer_state(self, checkpoint_folder: pathlib.Path):
        trainer_state_file = checkpoint_folder / Trainer.TRAINER_STATE_FILENAME

        with open(trainer_state_file, "r") as f:
            trainer_state = json.load(f)

        self.current_step = trainer_state["global_step"]
        self.current_epoch = trainer_state["epoch"]
        self.start_batch_idx = trainer_state["start_batch_idx"]

        for log_entry in trainer_state["log_history"]:
            self.metrics_data.append(
                [
                    log_entry["epoch"],
                    log_entry["step"],
                    log_entry["train_loss"],
                    log_entry["learning_rate"],
                    log_entry["grad_norm"],
                ]
            )

    def _maybe_save(
        self,
        batch_idx: int,
    ):
        if (
            self.args.save_steps is None
            or self.current_step % self.args.save_steps != 0
        ):
            return

        self.start_batch_idx = batch_idx + 1
        if self.start_batch_idx >= len(self.train_dataloader):
            self.start_batch_idx = 0

        model_folder = pathlib.Path(self.args.output_dir)
        checkpoint_folder = model_folder / f"checkpoint-{self.current_step}"

        if checkpoint_folder.exists() and not self.args.overwrite_output_dir:
            raise ValueError(f"{checkpoint_folder} already exists")

        checkpoint_folder.mkdir(parents=True, exist_ok=True)

        self._save_trainer_state(checkpoint_folder)
        self._save_training_args(checkpoint_folder)
        self._save_model(checkpoint_folder)
        self._save_optimizer(checkpoint_folder)
        self._save_scheduler(checkpoint_folder)
        self._save_scaler(checkpoint_folder)
        self._save_rng_state(checkpoint_folder)

    def _maybe_load_checkpoint(self, resume_from_checkpoint: str | int | bool):
        if isinstance(resume_from_checkpoint, bool):
            if not resume_from_checkpoint:
                return

            checkpoint_num = self._get_latest_checkpoint()
            if checkpoint_num == -1:
                return

        elif isinstance(resume_from_checkpoint, int):
            checkpoint_num = resume_from_checkpoint

        if isinstance(resume_from_checkpoint, str):
            checkpoint_folder = resume_from_checkpoint
        else:
            checkpoint_folder = self.args.output_dir / f"checkpoint-{checkpoint_num}"

        self._resume_from_checkpoint(checkpoint_folder)

    def _get_latest_checkpoint(self):
        files = os.listdir(self.args.output_dir)

        checkpoints = []
        for f in files:
            m = re.fullmatch(r"checkpoint-(\d+)", f)
            if m:
                checkpoints.append(int(m.group(1)))

        latest_checkpoint = max(checkpoints, default=-1)
        return latest_checkpoint

    def _resume_from_checkpoint(self, checkpoint_folder: pathlib.Path):
        self._load_trainer_state(checkpoint_folder)
        self._load_model(checkpoint_folder)
        self._load_optimizer(checkpoint_folder)
        self._load_scheduler(checkpoint_folder)
        self._load_scaler(checkpoint_folder)
        self._load_rng_state(checkpoint_folder)

    def train(self, resume_from_checkpoint: str | int | bool = False):
        self._maybe_load_checkpoint(resume_from_checkpoint)

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

        self.progress_task = self.progress.add_task(
            "Training",
            total=self.args.max_steps,
            completed=self.current_step,
        )

        self.live.start()
        self._refresh_log_metrics()

        self.model.to(self.device)

        try:
            accumulated_steps = 0

            while self.current_step < self.args.max_steps:
                self.epoch_task = self.epoch_progress.add_task(
                    f"Epoch {int(self.current_epoch + 1)}",
                    total=num_batch_per_epoch,
                    completed=self.start_batch_idx,
                )

                dataloader_iter = iter(self.train_dataloader)
                for _ in range(self.start_batch_idx):
                    next(dataloader_iter)

                for batch_idx, inputs in enumerate(
                    dataloader_iter, start=self.start_batch_idx
                ):

                    loss = self._training_step(inputs)
                    accumulated_steps += 1

                    self._update_loss_tracking(loss)

                    if accumulated_steps == self.args.gradient_accumulation_steps:
                        grad_norm = self._optimizer_step()
                        self._update_grad_norm_tracking(grad_norm)

                        self.current_step += 1
                        self.current_epoch = self.current_step / num_steps_per_epoch
                        accumulated_steps = 0

                        self._update_progress_and_log(
                            batch_idx=batch_idx,
                            total_batches=num_batch_per_epoch,
                        )

                        self._maybe_save(
                            batch_idx=batch_idx,
                        )

                        if self.current_step >= self.args.max_steps:
                            break

                if accumulated_steps > 0 and self.current_step < self.args.max_steps:
                    grad_norm = self._optimizer_step()
                    self._update_grad_norm_tracking(grad_norm)

                    self.current_step += 1
                    self.current_epoch = self.current_step / num_steps_per_epoch
                    accumulated_steps = 0

                    self._update_progress_and_log(
                        batch_idx=batch_idx,
                        total_batches=num_batch_per_epoch,
                    )

                    self._maybe_save(
                        batch_idx=batch_idx,
                    )

                self.start_batch_idx = 0
                self.epoch_progress.update(
                    self.epoch_task, completed=num_batch_per_epoch
                )
                self.epoch_progress.remove_task(self.epoch_task)

        finally:
            self.live.stop()

        self.console.print("\n[bold green]Training completed![/bold green]")
        self.console.print(f"[cyan]Total steps:[/cyan] {self.current_step}")
        self.console.print(f"[cyan]Epochs completed:[/cyan] {self.current_epoch}")
