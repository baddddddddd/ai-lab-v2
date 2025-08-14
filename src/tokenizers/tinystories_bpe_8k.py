import os

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

from .base_tokenizer import BaseTokenizer


class TinyStoriesBpe8kTokenizer(BaseTokenizer):
    SAVE_FILE = "./data/tokenizers/tinystories-bpe-8k.json"

    def __init__(
        self,
        max_length: int | None = None,
        padding: bool = False,
        return_overflowing_tokens: bool = False,
        stride: int = 0,
    ):
        super().__init__(max_length, padding, return_overflowing_tokens, stride)

        self.unk_token = "<unk>"
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"

        if os.path.exists(TinyStoriesBpe8kTokenizer.SAVE_FILE):
            self._load()
        else:
            print(f"Training TinyStoriesBpe8kTokenizer...")
            self.train()

    def train(self):
        tokenizer = Tokenizer(BPE(unk_token=self.unk_token))
        tokenizer.normalizer = NFKC()
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=8192,
            min_frequency=2,
            special_tokens=[self.unk_token, self.eos_token, self.pad_token],
            show_progress=True,
        )

        dataset = load_dataset("roneneldan/TinyStories", split="train")

        def batch_iterator(batch_size=1000):
            tok_dataset = dataset.select_columns("text")
            for batch in tok_dataset.iter(batch_size):
                yield batch["text"]

        tokenizer.train_from_iterator(
            iterator=batch_iterator(), trainer=trainer, length=len(dataset)
        )

        tokenizer.post_processor = TemplateProcessing(
            single=f"$A {self.eos_token}",
            special_tokens=[(self.eos_token, tokenizer.token_to_id(self.eos_token))],
        )

        os.makedirs(os.path.dirname(TinyStoriesBpe8kTokenizer.SAVE_FILE), exist_ok=True)
        tokenizer.save(TinyStoriesBpe8kTokenizer.SAVE_FILE)

        self._load()

    def _load(self):
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=TinyStoriesBpe8kTokenizer.SAVE_FILE,
            unk_token=self.unk_token,
            eos_token=self.eos_token,
            pad_token=self.pad_token,
        )

        self.vocab = self.tokenizer.get_vocab()

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        return self.tokenizer.tokenize(text, **kwargs)

    def _convert_id_to_token(self, token_id: int) -> str:
        return self.tokenizer.convert_ids_to_tokens([token_id])[0]

    def _convert_token_to_id(self, token: str) -> int:
        return self.tokenizer.convert_tokens_to_ids([token])[0]

    def _convert_tokens_to_string(self, tokens: list[str]) -> str:
        return self.tokenizer.convert_tokens_to_string(tokens)
