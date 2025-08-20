import multiprocessing
from pathlib import Path

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
        **tokenizer_kwargs,
    ):
        def tokenize(examples):
            tokens = tokenizer(examples[text_column], **tokenizer_kwargs)
            return tokens

        self.tokenizer = tokenizer
        self.raw_dataset = dataset
        self.text_column = text_column
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
                example_ids = torch.LongTensor(example["input_ids"])
                example_labels = torch.LongTensor(example["labels"])
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
            shuffle=shuffle,
            drop_last=True,
        )

    def dump_corpus(
        self,
        output_path: str | Path,
        separator: str = "\n\n",
        encoding: str = "utf-8",
        chunk_size: int = 1000,
        for_sentencepiece: bool = False,
    ):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding=encoding) as f:
            total_examples = len(self.raw_dataset)
            log_interval = chunk_size * 10
            total_written = 0

            for i in range(0, total_examples, chunk_size):
                end_idx = min(i + chunk_size, total_examples)
                chunk = self.raw_dataset[i:end_idx]

                if isinstance(chunk[self.text_column], list):
                    texts = chunk[self.text_column]
                else:
                    texts = [chunk[self.text_column]]

                for text in texts:
                    if for_sentencepiece:
                        lines = (
                            text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
                        )
                        for line in lines:
                            line = line.strip()
                            if line:
                                f.write(line + "\n")
                                total_written += 1
                    else:
                        f.write(text)
                        total_written += 1
                        if total_written < total_examples:
                            f.write(separator)

                if i % log_interval == 0:
                    print(f"Processed {i} examples...")

        print(f"Corpus dumped to {output_path}")
        print(f"Total examples/lines: {total_written}")
        print(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
        if for_sentencepiece:
            print("Formatted for SentencePiece training (one sentence per line)")


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

    def dump_corpus(
        self,
        output_path: str | Path,
        separator: str = "\n\n",
        encoding: str = "utf-8",
        max_examples: int | None = None,
        for_sentencepiece: bool = False,
    ):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding=encoding) as f:
            count = 0
            total_written = 0

            for example in self.raw_dataset:
                if max_examples and count >= max_examples:
                    break

                text = example[self.text_column]

                if for_sentencepiece:
                    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
                    for line in lines:
                        line = line.strip()
                        if line:
                            f.write(line + "\n")
                            total_written += 1
                else:
                    f.write(text)
                    f.write(separator)
                    total_written += 1

                count += 1
                if count % 10000 == 0:
                    print(f"Processed {count} examples...")

        print(f"Corpus dumped to {output_path}")
        print(f"Total examples/lines: {total_written}")
        print(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
        if for_sentencepiece:
            print("✓ Formatted for SentencePiece training (one sentence per line)")
