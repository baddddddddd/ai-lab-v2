import json
import os
import pathlib
import random
import re

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from ..datasets import BaseDataset
from ..models import BaseModel
from ..tokenizers import BaseTokenizer
from .base_trainer import BaseTrainer
from .training_config import TrainingConfig
from .schedulers import LinearWarmupCosineAnnealingLR


class CausalLmTrainer(BaseTrainer):
    TRAINER_STATE_FILENAME = "trainer_state.json"
    TRAINING_ARGS_FILENAME = "training_args.bin"
    OPTIMIZER_FILENAME = "optimizer.pt"
    SCHEDULER_FILENAME = "scheduler.pt"
    RNG_STATE_FILENAME = "rng_state.pth"

    def __init__(
        self,
        model: BaseModel,
        tokenizer: BaseTokenizer,
        args: TrainingConfig,
        train_dataset: BaseDataset,
        eval_dataset: BaseDataset | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=args.train_batch_size,
            drop_last=True,
            shuffle=True,
        )
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=args.betas,
            weight_decay=args.weight_decay,
        )
        self.scheduler = LinearWarmupCosineAnnealingLR(
            self.optimizer, args.warmup_steps, args.total_steps, args.min_lr
        )
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_token_id,
            label_smoothing=args.label_smoothing,
        )

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        print(f"Using device: {self.device}")

        self.model.to(self.device)

        os.makedirs(self.args.output_dir, exist_ok=True)

    def train(self, resume_from_checkpoint: os.PathLike[str] | bool = False):
        if resume_from_checkpoint:
            if isinstance(resume_from_checkpoint, bool):
                start_epoch, start_batch_idx, optimizer_steps = self.resume_latest()
            else:
                checkpoint_folder = pathlib.Path(resume_from_checkpoint).resolve()
                start_epoch, start_batch_idx, optimizer_steps = self.resume_from_folder(
                    checkpoint_folder
                )
        else:
            start_epoch, start_batch_idx, optimizer_steps = 0, 0, 0

        dataset_size = len(self.train_dataset)
        counter_width = len(str(dataset_size))

        scaler = torch.cuda.amp.GradScaler()

        self.model.train()
        for epoch in range(start_epoch, self.args.num_train_epochs):
            print(f"EPOCH {epoch + 1}")
            print(f"=" * 75)

            total_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(self.dataloader):
                if epoch <= start_epoch and batch_idx < start_batch_idx:
                    continue

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    logits = self.model(inputs)

                    logits = logits.view(-1, logits.size(-1))
                    targets = targets.view(-1)

                    loss = self.criterion(logits, targets)

                scaler.scale(loss).backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                self.scheduler.step()
                optimizer_steps += 1

                total_loss += loss.item()

                if self.args.save_steps and (
                    optimizer_steps % self.args.save_steps == 0
                ):
                    self.save_state(epoch, batch_idx, optimizer_steps)

                if self.args.logging_steps and (
                    optimizer_steps % self.args.logging_steps == 0
                ):
                    avg_loss = total_loss / self.args.logging_steps
                    processed = (batch_idx + 1) * self.args.train_batch_size
                    batch_progress = f"[{processed:>{counter_width}d}/{dataset_size}]"

                    avg_loss_log = f"{avg_loss:.6f}"
                    steps_log = f"{optimizer_steps}"
                    lr_log = f"{self.scheduler.get_last_lr()[0]:.8f}"

                    total_loss = 0.0
                    self.log(
                        batch_progress,
                        avg_loss=avg_loss_log,
                        lr=lr_log,
                        steps=optimizer_steps,
                    )

    def get_latest_checkpoint(self):
        files = os.listdir(self.args.output_dir)

        checkpoints = []
        for f in files:
            m = re.fullmatch(r"checkpoint-(\d+)", f)
            if m:
                checkpoints.append(int(m.group(1)))

        latest_checkpoint = max(checkpoints, default=0)
        return latest_checkpoint

    def resume_latest(self):
        latest_checkpoint = self.get_latest_checkpoint()
        if latest_checkpoint == 0:
            return 0, 0, 0
        else:
            foldername = f"checkpoint-{latest_checkpoint}"
            checkpoint_folder = self.args.output_dir / foldername
            return self.resume_from_folder(checkpoint_folder)

    def resume_from_folder(self, checkpoint_folder: pathlib.Path):
        self.model = self.model.__class__.from_pretrained(checkpoint_folder)
        self.load_optimizer(checkpoint_folder)
        self.load_scheduler(checkpoint_folder)
        self.load_rng_state(checkpoint_folder)

        trainer_state = self.load_trainer_state(checkpoint_folder)
        start_epoch = trainer_state["epoch"]
        start_batch_idx = trainer_state["batch_idx"]
        optimizer_steps = trainer_state["optimizer_steps"]
        return start_epoch, start_batch_idx, optimizer_steps

    def load_trainer_state(self, checkpoint_folder: pathlib.Path) -> dict:
        json_file = checkpoint_folder / CausalLmTrainer.TRAINER_STATE_FILENAME
        with open(json_file, "r") as f:
            json_string = f.read()
            state_dict = json.loads(json_string)

        return state_dict

    def save_trainer_state(self, save_directory, epoch, batch_idx, optimizer_steps):
        state_dict = {
            "epoch": epoch,
            "batch_idx": batch_idx + 1,
            "optimizer_steps": optimizer_steps,
        }

        json_string = json.dumps(state_dict)
        save_path = save_directory / CausalLmTrainer.TRAINER_STATE_FILENAME

        with open(save_path, "w") as f:
            f.write(json_string)

    def load_optimizer(self, checkpoint_folder: pathlib.Path):
        optimizer_file = checkpoint_folder / CausalLmTrainer.OPTIMIZER_FILENAME
        optimizer_state = torch.load(optimizer_file)
        self.optimizer.load_state_dict(optimizer_state)

    def save_optimizer(self, save_directory: pathlib.Path):
        optimizer_state = self.optimizer.state_dict()
        optimizer_file = save_directory / CausalLmTrainer.OPTIMIZER_FILENAME
        torch.save(optimizer_state, optimizer_file)

    def load_scheduler(self, checkpoint_folder: pathlib.Path):
        scheduler_file = checkpoint_folder / CausalLmTrainer.SCHEDULER_FILENAME
        scheduler_state = torch.load(scheduler_file)
        self.scheduler.load_state_dict(scheduler_state)

    def save_scheduler(self, save_directory: pathlib.Path):
        scheduler_state = self.scheduler.state_dict()
        scheduler_file = save_directory / CausalLmTrainer.SCHEDULER_FILENAME
        torch.save(scheduler_state, scheduler_file)

    def load_rng_state(self, checkpoint_folder: pathlib.Path):
        trainer_state_file = checkpoint_folder / CausalLmTrainer.RNG_STATE_FILENAME
        trainer_state = torch.load(trainer_state_file, weights_only=False)

        torch.set_rng_state(trainer_state["torch_rng_state"])
        torch.cuda.set_rng_state_all(trainer_state["cuda_rng_state"])
        np.random.set_state(trainer_state["numpy_rng_state"])
        random.setstate(trainer_state["python_rng_state"])

        if trainer_state["dataloader_state"] and hasattr(
            self.dataloader.sampler, "load_state_dict"
        ):
            self.dataloader.sampler.load_state_dict(trainer_state["dataloader_state"])

    def save_rng_state(self, save_directory: pathlib.Path):
        rng_state = {
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "dataloader_state": (
                self.dataloader.sampler.state_dict()
                if hasattr(self.dataloader.sampler, "state_dict")
                else None
            ),
        }

        save_path = save_directory / CausalLmTrainer.RNG_STATE_FILENAME
        torch.save(rng_state, save_path)

    def save_state(self, epoch, batch_idx, optimizer_steps):
        checkpoint_folder = f"checkpoint-{optimizer_steps}"
        save_folder = self.args.output_dir / checkpoint_folder

        if os.path.exists(save_folder):
            raise ValueError(f"{save_folder} folder already exists")

        self.model.save_pretrained(save_directory=save_folder)
        self.save_trainer_state(save_folder, epoch, batch_idx, optimizer_steps)
        self.save_optimizer(save_directory=save_folder)
        self.save_scheduler(save_directory=save_folder)
        self.save_rng_state(save_directory=save_folder)

        print(f"[INFO] Saved {save_folder}")

    def log(self, *args, **kwargs):
        args_string = " ".join(args)
        kwargs_string = " ".join([f"{k}={v}" for k, v in kwargs.items()])

        log_string = f"{args_string} {kwargs_string}"
        print(log_string)
