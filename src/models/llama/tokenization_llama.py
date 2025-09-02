import re

import sentencepiece as spm
import torch

from ...tokenizers import BaseTokenizer

SPIECE_UNDERLINE = "▁"


class LlamaTokenizer(BaseTokenizer):
    def __init__(
        self,
        vocab_file: str,
        unk_token: str = "<unk>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        add_bos_token: bool = True,
        add_eos_token: bool = False,
    ):

        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token

        self.tokenizer = spm.SentencePieceProcessor(model_file=vocab_file)

        self.vocab = {
            self.tokenizer.id_to_piece(id_): id_
            for id_ in range(self.tokenizer.get_piece_size())
        }

        self.special_tokens = set(
            [tok for tok in [unk_token, bos_token, eos_token] if tok is not None]
        )

        pattern = "(" + "|".join((re.escape(tok) for tok in self.special_tokens)) + ")"
        self.split_token_re = re.compile(pattern)

        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
        )

    @staticmethod
    def train(
        corpus_file: str,
        model_prefix: str = "tokenizer",
        vocab_size: str = 32000,
        character_coverage: float = 0.99995,
        add_dummy_prefix: bool = False,
        unk_token: str = "<unk>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
    ):
        import multiprocessing

        print("Training LLaMa Tokenizer...")
        spm.SentencePieceTrainer.train(
            input=corpus_file,
            model_prefix=model_prefix,
            model_type="bpe",
            vocab_size=vocab_size,
            unk_piece=unk_token,
            bos_piece=bos_token,
            eos_piece=eos_token,
            self_test_sample_size=0,
            input_format="text",
            character_coverage=character_coverage,
            num_threads=multiprocessing.cpu_count(),
            split_digits=True,
            allow_whitespace_only_pieces=True,
            byte_fallback=True,
            unk_surface=" \uFFF7 ",
            normalization_rule_name="identity",
            add_dummy_prefix=add_dummy_prefix,
        )
        print(f"Finished training LLaMa Tokenizer")

    def get_vocab(self) -> dict[str, int]:
        return self.vocab

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        parts = self.split_token_re.split(text)

        tokens = []
        for part in parts:
            if part in self.special_tokens:
                tokens.append(part)
            else:
                tokens.extend(self.tokenizer.encode_as_pieces(part))

        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self.tokenizer.piece_to_id(token)

    def _convert_id_to_token(self, token_id: int) -> str:
        return self.tokenizer.id_to_piece(token_id)

    def _convert_tokens_to_string(self, tokens: list[str]) -> str:
        out = ""

        sub_tokens = []
        for tok in tokens:
            if tok in self.special_tokens:
                out += self.tokenizer.decode(sub_tokens) + tok
                sub_tokens.clear()
            else:
                if not sub_tokens and tok.startswith(SPIECE_UNDERLINE):
                    out += " "

                sub_tokens.append(tok)

        out += self.tokenizer.decode(sub_tokens)
        return out

    def build_inputs_with_special_tokens(self, token_ids: list[int]) -> list[int]:
        prepend = [self.bos_token_id] if self.add_bos_token else []
        append = [self.eos_token_id] if self.add_eos_token else []

        return prepend + token_ids + append

    def num_special_tokens_to_add(self):
        return int(self.add_bos_token) + int(self.add_eos_token)
