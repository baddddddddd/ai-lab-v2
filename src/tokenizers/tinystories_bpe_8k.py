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

    def __init__(self, add_eos_token: bool = True):
        UNK_TOKEN = "<unk>"
        EOS_TOKEN = "<eos>"
        PAD_TOKEN = "<pad>"

        self.add_eos_token = add_eos_token

        if os.path.exists(TinyStoriesBpe8kTokenizer.SAVE_FILE):
            self._load(
                unk_token=UNK_TOKEN,
                eos_token=EOS_TOKEN,
                pad_token=PAD_TOKEN,
            )
        else:
            print(f"Training TinyStoriesBpe8kTokenizer...")
            self.train(
                unk_token=UNK_TOKEN,
                eos_token=EOS_TOKEN,
                pad_token=PAD_TOKEN,
            )

        super().__init__(
            unk_token=UNK_TOKEN,
            eos_token=EOS_TOKEN,
            pad_token=PAD_TOKEN,
        )

    def train(self, unk_token: str, eos_token: str, pad_token: str):
        tokenizer = Tokenizer(BPE(unk_token=unk_token))
        tokenizer.normalizer = NFKC()
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=8192,
            min_frequency=2,
            special_tokens=[unk_token, eos_token, pad_token],
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
            single=f"$A {eos_token}",
            special_tokens=[(eos_token, tokenizer.token_to_id(eos_token))],
        )

        os.makedirs(os.path.dirname(TinyStoriesBpe8kTokenizer.SAVE_FILE), exist_ok=True)
        tokenizer.save(TinyStoriesBpe8kTokenizer.SAVE_FILE)

        self._load(
            unk_token=unk_token,
            eos_token=eos_token,
            pad_token=pad_token,
        )

    def _load(self, unk_token: str, eos_token: str, pad_token: str):
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=TinyStoriesBpe8kTokenizer.SAVE_FILE,
            unk_token=unk_token,
            eos_token=eos_token,
            pad_token=pad_token,
        )

        self.vocab = self.tokenizer.get_vocab()

    def get_vocab(self) -> dict[str, int]:
        return self.vocab

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        return self.tokenizer.tokenize(text, **kwargs)

    def _convert_id_to_token(self, token_id: int) -> str:
        return self.tokenizer.convert_ids_to_tokens([token_id])[0]

    def _convert_token_to_id(self, token: str) -> int:
        return self.tokenizer.convert_tokens_to_ids([token])[0]

    def _convert_tokens_to_string(self, tokens: list[str]) -> str:
        return self.tokenizer.convert_tokens_to_string(tokens)

    def build_inputs_with_special_tokens(self, token_ids: list[int]) -> list[int]:
        if self.add_eos_token:
            return token_ids + [self.eos_token_id]
        else:
            return token_ids

    def num_special_tokens_to_add(self):
        return 1 if self.add_eos_token else 0
