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

        self.model.to(self.device)

    def get_dataset(self, split: str = "test"):
        return load_dataset("dlwh/wikitext_2_detokenized", split=split)

    def tokenize_dataset(self, dataset: Dataset, max_length: int):
        encodings = self.tokenizer.encode(
            "\n\n".join(dataset["text"]), return_tensors="pt"
        )
        return encodings

    @torch.no_grad()
    def evaluate(self, max_length: int, split: str = "test"):
        self.model.eval()

        dataset = self.get_dataset(split=split)
        encodings = self.tokenize_dataset(dataset, max_length=max_length)
        nll_sum = 0.0
        n_tokens = 0

        for i in tqdm(
            range(0, encodings.size(1), max_length), desc="WikiText-2 Perplexity"
        ):
            input_ids = encodings[:, i : i + max_length + 1].to(self.device)
            labels = input_ids.clone()

            input_ids = input_ids[..., :-1]
            labels = labels[..., 1:]

            outputs = self.model(input_ids)

            logits = outputs.logits
            logits = logits.reshape(-1, logits.size(-1))
            labels = labels.reshape(-1)

            loss_per_token = F.cross_entropy(
                logits,
                labels,
                reduction="none",
                ignore_index=-100,
            )
            nll_sum += loss_per_token.sum().item()

            num_loss_tokens = (labels != -100).sum().item()
            n_tokens += num_loss_tokens

        avg_loss = nll_sum / n_tokens
        ppl = math.exp(avg_loss)
        return ppl
