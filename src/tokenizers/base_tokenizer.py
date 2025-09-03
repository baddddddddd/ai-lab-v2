import itertools
import re
import string

import jinja2
import torch


class BaseTokenizer:
    def __init__(
        self,
        unk_token: str | None = None,
        bos_token: str | None = None,
        eos_token: str | None = None,
        pad_token: str | None = None,
        sep_token: str | None = None,
        cls_token: str | None = None,
        mask_token: str | None = None,
        chat_template: str | None = None,
    ):
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.sep_token = sep_token
        self.cls_token = cls_token
        self.mask_token = mask_token
        self.chat_template = chat_template

        self.unk_token_id = self._convert_token_to_id(unk_token) if unk_token else None
        self.bos_token_id = self._convert_token_to_id(bos_token) if bos_token else None
        self.eos_token_id = self._convert_token_to_id(eos_token) if eos_token else None
        self.pad_token_id = self._convert_token_to_id(pad_token) if pad_token else None
        self.sep_token_id = self._convert_token_to_id(sep_token) if sep_token else None
        self.cls_token_id = self._convert_token_to_id(cls_token) if cls_token else None
        self.mask_token_id = (
            self._convert_token_to_id(mask_token) if mask_token else None
        )

        if self.chat_template is None:
            self.chat_template = "{{ bos_token if bos_token else '' }}{%- for message in messages %}{{ '' if loop.first else '\n' }}### {{ message['role'] | title }}:\n{{ message['content'].strip() }}{{ eos_token if eos_token and message['role'] == 'assistant' else '' }}{% endfor -%}{{ '\n### Assistant:' if add_generation_prompt else '' }}"

    def __call__(
        self,
        texts: list[str],
        add_special_tokens: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        stride: int = 0,
        return_tensors: str | None = None,
        return_overflowing_tokens: bool = False,
        **kwargs,
    ) -> list[list[int] | torch.Tensor]:
        input_ids = []
        for text in texts:
            ids = self.encode(
                text,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                return_tensors=return_tensors,
                return_overflowing_tokens=return_overflowing_tokens,
                **kwargs,
            )
            if return_tensors is None and return_overflowing_tokens:
                input_ids += ids
            else:
                input_ids.append(ids)

        if return_tensors == "pt":
            input_ids = torch.cat(input_ids)

        encoded = {
            "input_ids": input_ids,
        }
        return encoded

    def get_vocab(self) -> dict[str, int]:
        raise NotImplementedError("get_vocab() is not implemented")

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError("vocab_size is not implemented")

    def _tokenize(self, text: str, **kwargs) -> list[str]:
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

    def prepare_for_tokenization(self, text, **kwargs):
        return (text, kwargs)

    def tokenize(self, text: str, **kwargs) -> list[str]:
        return self._tokenize(text, **kwargs)

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [self._convert_token_to_id(tok) for tok in tokens]

    def convert_ids_to_tokens(self, ids: list[int] | torch.Tensor) -> list[str]:
        tokens = [self._convert_id_to_token(id_) for id_ in ids]
        return tokens

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return self._convert_tokens_to_string(tokens)

    def _encode(self, text: str, **kwargs) -> list[int]:
        tokens = self.tokenize(text, **kwargs)
        ids = self.convert_tokens_to_ids(tokens)
        return ids

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        stride: int = 0,
        return_tensors: str | None = None,
        return_overflowing_tokens: bool = False,
        **kwargs,
    ) -> list[int] | list[list[int]] | torch.Tensor:
        all_kwargs = {
            "add_special_tokens": add_special_tokens,
            "padding": padding,
            "truncation": truncation,
            "max_length": max_length,
            "stride": stride,
            "return_tensors": return_tensors,
            "return_overflowing_tokens": return_overflowing_tokens,
            **kwargs,
        }

        text, kwargs = self.prepare_for_tokenization(text, **all_kwargs)
        ids = self._encode(text, **kwargs)

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
                steps = adjusted_max_length - stride
                ids = [
                    ids[i : i + adjusted_max_length]
                    for i in range(0, len(ids) - stride, steps)
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

            if self.pad_token_id is None:
                raise ValueError(
                    "padding is set to True, but no pad_token_id was provided."
                )

            if return_overflowing_tokens:
                for i in range(len(ids)):
                    pad_len = max_length - len(ids[i])
                    ids[i] += [self.pad_token_id] * pad_len

            else:
                pad_len = max_length - len(ids)
                ids += [self.pad_token_id] * pad_len

        if return_tensors == "pt":
            return torch.LongTensor([ids])
        else:
            return ids

    def _decode(self, token_ids: list[int]) -> str:
        tokens = self.convert_ids_to_tokens(token_ids)
        s = self.convert_tokens_to_string(tokens)
        return s

    def decode(self, token_ids: list[int] | torch.Tensor) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        return self._decode(token_ids)

    def _batch_decode(self, sequences: list[list[int]]) -> list[str]:
        decoded = [self._decode(seq) for seq in sequences]
        return decoded

    def batch_decode(self, sequences: list[list[int] | torch.Tensor]) -> list[str]:
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()

        return self._batch_decode(sequences)

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]] | list[list[dict[str, str]]],
        chat_template: str | None = None,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        return_tensors: str | None = None,
        **tokenizer_kwargs,
    ):
        if isinstance(conversation, list) and isinstance(conversation[0], dict):
            is_batched = False
            conversation_list = [conversation]
        else:
            is_batched = True
            conversation_list = conversation

        template = jinja2.Template(
            self.chat_template if chat_template is None else chat_template
        )

        formatted_list = []
        for conversation in conversation_list:
            formatted = template.render(
                messages=conversation,
                add_generation_prompt=add_generation_prompt,
                # might be better to create a dict of special tokens
                unk_token=self.unk_token,
                bos_token=self.bos_token,
                eos_token=self.eos_token,
                pad_token=self.pad_token,
                cls_token=self.cls_token,
                sep_token=self.sep_token,
                mask_token=self.mask_token,
            )

            formatted_list.append(formatted)

        if not tokenize:
            return formatted_list if is_batched else formatted_list[0]

        encoded = self(
            formatted_list,
            return_tensors=return_tensors,
            **tokenizer_kwargs,
        )["input_ids"]

        if return_tensors == "pt":
            return encoded

        return encoded if is_batched else encoded[0]
