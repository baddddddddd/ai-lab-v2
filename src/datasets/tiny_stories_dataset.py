import os

import datasets
import torch
from typing import Sized

from .base_dataset import BaseDataset
from ..tokenizers import BaseTokenizer


class TinyStoriesDataset(BaseDataset):
    def __init__(
        self,
        split: str,
        tokenizer: BaseTokenizer,
        **kwargs,
    ):
        raw_dataset = datasets.load_dataset(
            "roneneldan/TinyStories", split=split, **kwargs
        )

        def encode(example):
            texts = example["text"]
            encoded = tokenizer(texts)
            return {"input_ids": encoded}

        if tokenizer.max_length is not None:
            tokenizer.max_length += 1
            tokenizer.stride += 1

        num_proc = os.cpu_count()
        self.dataset = raw_dataset.map(
            encode,
            remove_columns=["text"],
            batched=True,
            num_proc=num_proc,
        )

        if tokenizer.max_length is not None:
            tokenizer.max_length -= 1
            tokenizer.stride -= 1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        token_ids = row["input_ids"]

        inputs = torch.LongTensor(token_ids[:-1])
        targets = torch.LongTensor(token_ids[1:])

        return inputs, targets
