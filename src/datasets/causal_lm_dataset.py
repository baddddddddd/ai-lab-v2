import multiprocessing
from pathlib import Path

import spacy
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
        return_position_ids: bool = False,
        packed: bool = False,
        packed_length: int | None = None,
        packing_stride: int = 0,
        **tokenizer_kwargs,
    ):
        self.tokenizer = tokenizer
        self.raw_dataset = dataset
        self.text_column = text_column
        self.return_position_ids = return_position_ids

        if packed:
            assert (
                packed_length is not None
            ), "packed_length must be specified when packed=True"
            assert packing_stride >= 0, "packing_stride must be non-negative"
            assert (
                packing_stride < packed_length
            ), "packing_stride must be less than packed_length"

            self.dataset = self._create_packed_dataset(
                packed_length=packed_length,
                packing_stride=packing_stride,
                return_position_ids=return_position_ids,
                **tokenizer_kwargs,
            )
        else:
            self.dataset = self._create_tokenized_dataset(**tokenizer_kwargs)

    def _create_tokenized_dataset(self, **tokenizer_kwargs):
        def tokenize(examples):
            return self.tokenizer(examples[self.text_column], **tokenizer_kwargs)

        return self.raw_dataset.map(
            tokenize,
            remove_columns=self.text_column,
            batched=True,
            num_proc=multiprocessing.cpu_count(),
        )

    def _create_packed_dataset(
        self,
        packed_length: int,
        packing_stride: int = 0,
        return_position_ids: bool = True,
        **tokenizer_kwargs,
    ):
        tokenized_dataset = self._create_tokenized_dataset(**tokenizer_kwargs)
        sequences, position_ids = self._pack_sequences(
            tokenized_dataset,
            packed_length=packed_length,
            packing_stride=packing_stride,
            return_position_ids=return_position_ids,
        )

        dataset_dict = {}
        dataset_dict["input_ids"] = sequences

        if return_position_ids:
            dataset_dict["position_ids"] = position_ids

        return HFDataset.from_dict(dataset_dict)

    def _pack_sequences(
        self,
        tokenized_dataset,
        packed_length: int,
        packing_stride: int = 0,
        return_position_ids: bool = True,
    ):
        sequences = []
        position_ids = []
        cur_seq = []
        cur_pos = []

        for item in tokenized_dataset:
            input_ids = item["input_ids"]
            cur_seq.extend(input_ids)

            if return_position_ids:
                cur_pos.extend(range(len(input_ids)))

            while len(cur_seq) >= packed_length:
                sequences.append(cur_seq[:packed_length])
                cur_seq = cur_seq[packed_length - packing_stride :]

                if return_position_ids:
                    position_ids.append(cur_pos[:packed_length])
                    cur_pos = cur_pos[packed_length - packing_stride :]

        return sequences, position_ids

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx].copy()
        ids = example["input_ids"]

        example["input_ids"] = ids[:-1]
        example["labels"] = ids[1:]

        if self.return_position_ids:
            example["position_ids"].pop()

        return example

    def get_dataloader(self, batch_size: int, shuffle: bool = True):
        def collate_fn(examples):
            input_ids = []
            labels = []
            position_ids = []

            for example in examples:
                example_ids = torch.LongTensor(example["input_ids"])
                example_labels = torch.LongTensor(example["labels"])
                if self.tokenizer.pad_token_id:
                    example_labels[example_labels == self.tokenizer.pad_token_id] = -100

                input_ids.append(example_ids)
                labels.append(example_labels)

                if self.return_position_ids:
                    position_ids.append(torch.LongTensor(example["position_ids"]))

            batch = dict()
            batch["input_ids"] = torch.stack(input_ids)
            batch["labels"] = torch.stack(labels)

            if self.return_position_ids:
                batch["position_ids"] = torch.stack(position_ids)

            return batch

        return DataLoader(
            self,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=shuffle,
            num_workers=min(8, multiprocessing.cpu_count()),
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
        )


class CausalLmStreamingDataset(BaseDataset):
    def __init__(
        self,
        dataset: HFIterableDataset,
        tokenizer: BaseTokenizer,
        text_column: str = "text",
        **tokenizer_kwargs,
    ):

        self.tokenizer = tokenizer
        self.raw_dataset = dataset
        self.text_column = text_column
        self.tokenizer_kwargs = tokenizer_kwargs

    def __iter__(self):
        return_overflowing_tokens = self.tokenizer_kwargs.get(
            "return_overflowing_tokens", False
        )

        for example in self.raw_dataset:
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
                example_ids = torch.LongTensor(example["input_ids"])
                example_labels = torch.LongTensor(example["labels"]).clone()
                if self.tokenizer.pad_token_id:
                    example_labels[example_labels == self.tokenizer.pad_token_id] = -100

                input_ids.append(example_ids)
                labels.append(example_labels)

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
