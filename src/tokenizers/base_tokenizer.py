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
        self,
        texts: list[str],
        add_special_tokens: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        return_overflowing_tokens: bool = False,
    ) -> list[list[int] | torch.Tensor]:
        input_ids = []
        for text in texts:
            ids = self.encode(
                text,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
                return_overflowing_tokens=return_overflowing_tokens,
            )
            if isinstance(ids[0], list):
                input_ids += ids
            else:
                input_ids.append(ids)

        if return_tensors == "pt":
            if return_overflowing_tokens:
                input_ids = torch.cat(input_ids)
            else:
                input_ids = torch.stack(input_ids)

        encoded = {
            "input_ids": input_ids,
        }
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

    def build_inputs_with_special_tokens(self, token_ids: list[int]) -> list[int]:
        raise NotImplementedError(
            "build_inputs_with_special_tokens() method is not implemented"
        )

    def num_special_tokens_to_add(self):
        raise NotImplementedError(
            "num_special_tokens_to_add() method is not implemented"
        )

    def tokenize(self, text: str) -> list[str]:
        return self._tokenize(text)

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [self._convert_token_to_id(tok) for tok in tokens]

    def convert_ids_to_tokens(self, ids: list[int] | torch.Tensor) -> list[str]:
        tokens = [self._convert_id_to_token(id_) for id_ in ids]
        return tokens

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return self._convert_tokens_to_string(tokens)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        return_overflowing_tokens: bool = False,
    ) -> list[int] | list[list[int]] | torch.Tensor:
        tokens = self.tokenize(text)
        ids = self.convert_tokens_to_ids(tokens)

        if truncation:
            if max_length is None:
                raise ValueError(
                    "truncation is set to True, but no max_length was provided."
                )

            if add_special_tokens:
                adjusted_max_length = max_length - self.num_special_tokens_to_add()
            else:
                adjusted_max_length = max_length

            if return_overflowing_tokens:
                ids = [
                    ids[i : i + adjusted_max_length]
                    for i in range(0, len(ids), adjusted_max_length)
                ]
            else:
                ids = ids[:adjusted_max_length]

        if add_special_tokens:
            if return_overflowing_tokens:
                ids = [self.build_inputs_with_special_tokens(id_) for id_ in ids]
            else:
                ids = self.build_inputs_with_special_tokens(ids)

        if padding:
            if max_length is None:
                raise ValueError(
                    "padding is set to True, but no max_length was provided."
                )

            if return_overflowing_tokens:
                for i in range(len(ids)):
                    pad_len = max_length - len(ids[i])
                    ids[i] += [self.pad_token_id] * pad_len

            else:
                pad_len = max_length - len(ids)
                ids += [self.pad_token_id] * pad_len

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
