import itertools
import re
import string

import torch


class BaseTokenizer:
    def __init__(
        self,
        max_length: int | None = None,
        padding: bool = False,
        return_overflowing_tokens: bool = False,
        stride: int = 0,
    ):
        self.vocab = {}
        self.unk_token = ""
        self.eos_token = ""
        self.pad_token = ""

        self.max_length = max_length
        self.padding = padding
        self.return_overflowing_tokens = return_overflowing_tokens
        self.stride = stride

    @property
    def unk_token_id(self):
        return self._convert_token_to_id(self.unk_token)

    @property
    def eos_token_id(self):
        return self._convert_token_to_id(self.eos_token)

    @property
    def pad_token_id(self):
        return self._convert_token_to_id(self.pad_token)

    def __call__(
        self, texts: list[str], return_tensors: str | None = None
    ) -> list[list[int] | torch.Tensor]:
        encoded = list(
            itertools.chain.from_iterable(
                [self.encode(text, return_tensors=return_tensors) for text in texts]
            )
        )
        if return_tensors == "pt":
            return torch.stack(encoded)
        else:
            return encoded

    def _tokenize(self, text: str) -> list[str]:
        raise NotImplementedError("_tokenize() method is not implemented")

    def _convert_token_to_id(self, token: str) -> int:
        raise NotImplementedError("_convert_token_to_id() method is not implemented")

    def _convert_id_to_token(self, token_id: int) -> str:
        raise NotImplementedError("_convert_id_to_token() method is not implemented")

    def _convert_tokens_to_string(self, tokens: list[str]) -> str:
        raise NotImplementedError(
            "_convert_tokens_to_string() method is not implemented"
        )

    def tokenize(self, text: str) -> list[list[str]]:
        full_tokens = self._tokenize(text)
        full_tokens.append(self.eos_token)

        if self.max_length is None:
            return [full_tokens]

        res = []
        for i in range(
            0,
            len(full_tokens) if self.return_overflowing_tokens else 1,
            self.max_length - self.stride,
        ):
            tokens = full_tokens[i : i + self.max_length]
            res.append(tokens)

        if self.padding:
            pads = [self.pad_token] * (self.max_length - len(res[-1]))
            res[-1] += pads

        return res

    def convert_tokens_to_ids(self, tokens_list: list[list[str]]) -> list[list[int]]:
        res = []
        for tokens in tokens_list:
            ids = [self._convert_token_to_id(token) for token in tokens]
            res.append(ids)

        return res

    def convert_ids_to_tokens(self, ids: list[int] | torch.Tensor) -> list[str]:
        tokens = [self._convert_id_to_token(id_) for id_ in ids]
        return tokens

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return self._convert_tokens_to_string(tokens)

    def encode(
        self, text: str, return_tensors: str | None = None
    ) -> list[int] | torch.Tensor:
        tokens = self.tokenize(text)
        ids = self.convert_tokens_to_ids(tokens)

        if return_tensors == "pt":
            return torch.LongTensor(ids)
        else:
            return ids

    def decode(self, token_ids: list[int] | torch.Tensor) -> str:
        tokens = self.convert_ids_to_tokens(token_ids)
        s = self.convert_tokens_to_string(tokens)
        return s

    def batch_decode(self, sequences: list[list[int] | torch.Tensor]) -> list[str]:
        decoded = [self.decode(seq) for seq in sequences]
        return decoded

    def get_vocab(self) -> dict[str, int]:
        return self.vocab

    def get_vocab_size(self) -> int:
        return len(self.vocab)
