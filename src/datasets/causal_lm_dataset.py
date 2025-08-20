import multiprocessing

import torch
from torch.utils.data import DataLoader
from datasets import Dataset as HFDataset, IterableDataset as HFIterableDataset

from ..tokenizers import BaseTokenizer
from .base_dataset import BaseDataset, BaseStreamingDataset


class CausalLmDataset(BaseDataset):
    def __init__(
        self,
        dataset: HFDataset,
        tokenizer: BaseTokenizer,
        text_column: str = "text",
        **tokenizer_kwargs
    ):
        def tokenize(examples):
            tokens = tokenizer(examples[text_column], **tokenizer_kwargs)
            return tokens

        self.dataset = dataset.map(
            tokenize,
            remove_columns=text_column,
            batched=True,
            num_proc=multiprocessing.cpu_count(),
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ids = self.dataset[idx]["input_ids"]
        return {
            "input_ids": ids[:-1],
            "labels": ids[1:],
        }

    def get_dataloader(self, batch_size: int, shuffle: bool = False):
        def collate_fn(examples):
            input_ids = []
            labels = []

            for example in examples:
                input_ids.append(torch.LongTensor(example["input_ids"]))
                labels.append(torch.LongTensor(example["labels"]))

            return {
                "input_ids": torch.stack(input_ids),
                "labels": torch.stack(labels),
            }

        return DataLoader(
            self,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=shuffle,
            drop_last=True,
        )


class CausalLmStreamingDataset(BaseStreamingDataset):
    def __init__(
        self,
        dataset: HFIterableDataset,
        tokenizer: BaseTokenizer,
        text_column: str = "text",
        **tokenizer_kwargs
    ):

        self.tokenizer = tokenizer
        self.dataset = dataset
        self.text_column = text_column
        self.tokenizer_kwargs = tokenizer_kwargs

    def __iter__(self):
        return_overflowing_tokens = self.tokenizer_kwargs.get(
            "return_overflowing_tokens", False
        )

        for example in self.dataset:
            ids = self.tokenizer.encode(
                example[self.text_column], **self.tokenizer_kwargs
            )
            if return_overflowing_tokens:
                for id_ in ids:
                    yield {
                        "input_ids": id_[:-1],
                        "labels": id_[1:],
                    }
            else:
                yield {
                    "input_ids": ids[:-1],
                    "labels": ids[1:],
                }

    def get_dataloader(self, batch_size: int):
        def collate_fn(examples):
            input_ids = []
            labels = []

            for example in examples:
                input_ids.append(torch.LongTensor(example["input_ids"]))
                labels.append(torch.LongTensor(example["labels"]))

            return {
                "input_ids": torch.stack(input_ids),
                "labels": torch.stack(labels),
            }

        return DataLoader(
            self,
            batch_size=batch_size,
            collate_fn=collate_fn,
            drop_last=True,
        )
