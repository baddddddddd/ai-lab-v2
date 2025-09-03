import math

import torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from tqdm import tqdm

from ..models import BaseModel
from ..tokenizers import BaseTokenizer


class WikiText2PerplexityEvaluator:
    def __init__(
        self,
        model: BaseModel,
        tokenizer: BaseTokenizer,
        device: str | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if self.device is None:
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        self.model.to(device)

    def get_dataset(self, split: str = "validation"):
        return load_dataset("dlwh/wikitext_2_detokenized", split=split)

    def tokenize_dataset(self, dataset: Dataset, max_length: int):
        def tokenize_fn(examples):
            return self.tokenizer(
                examples["text"],
                padding=True,
                truncation=True,
                max_length=max_length + 1,
                stride=1,
                return_overflowing_tokens=True,
            )

        return dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    @torch.no_grad()
    def evaluate(self, max_length: int, batch_size: int, split: str = "validation"):
        self.model.eval()

        dataset = self.get_dataset(split=split)
        tokenized_dataset = self.tokenize_dataset(dataset, max_length=max_length)

        total_loss = 0.0
        total_tokens = 0

        iterator = tqdm(
            range(0, len(tokenized_dataset), batch_size), desc="WikiText-2 Perplexity"
        )
        for i in iterator:
            batch = tokenized_dataset.select(
                range(i, min(len(tokenized_dataset), i + batch_size))
            ).to_dict()
            input_ids = torch.LongTensor(batch["input_ids"], device=self.device)
            labels = input_ids.clone()

            input_ids = input_ids[..., :-1]
            labels = labels[..., 1:]

            labels[labels == self.tokenizer.pad_token_id] = -100

            output = self.model(
                input_ids=input_ids,
                labels=labels,
            )

            logits = output.logits

            x = logits.reshape(-1, logits.size(-1))
            y = labels.reshape(-1)

            loss_per_token = F.cross_entropy(x, y, reduction="none", ignore_index=-100)

            total_loss += loss_per_token.sum().item()
            total_tokens += (labels != -100).sum().item()

        avg_loss = total_loss / total_tokens
        ppl = math.exp(avg_loss)
        return ppl
