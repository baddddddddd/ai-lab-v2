import re
import string
import unicodedata

from .base_tokenizer import BaseTokenizer


class StrippedAsciiTokenizer(BaseTokenizer):
    TRANSLATION_TABLE = str.maketrans(
        {
            "“": '"',
            "”": '"',
            "‘": "'",
            "’": "'",
            "–": "-",
            "—": "-",
            "−": "-",
            "‐": "-",
            "×": "x",
            "÷": "/",
            "\u00A0": " ",  # non-breaking space
            "\u202F": " ",  # narrow NBSP
            "\u200B": None,
            "\u200C": None,
            "\u200D": None,  # zero-width stuff
        }
    )

    def __init__(
        self,
        max_length: int | None = None,
        padding: bool = False,
        return_overflowing_tokens: bool = False,
        stride: int = 0,
    ):
        super().__init__(max_length, padding, return_overflowing_tokens, stride)

        self.charset = string.digits + string.ascii_letters + string.punctuation
        self.whitespace = " \n\t"
        self.unk_token = "\x00"
        self.eos_token = "\x01"
        self.pad_token = "\x02"

        self.id_to_token = (
            self.charset
            + self.whitespace
            + self.unk_token
            + self.eos_token
            + self.pad_token
        )

        self.vocab = dict(zip(self.id_to_token, range(len(self.id_to_token))))

        pattern = "|".join(re.escape(c) for c in self.id_to_token)
        self.re = re.compile(pattern)

    def _normalize_to_ascii(self, text: str) -> str:
        text = text.translate(StrippedAsciiTokenizer.TRANSLATION_TABLE)

        text = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")

        text = text.encode("ascii", "ignore").decode("ascii")
        return text

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        normalized = self._normalize_to_ascii(text)
        tokens = [c if c in self.vocab else self.unk_token for c in normalized]
        return tokens

    def _convert_id_to_token(self, token_id: int) -> str:
        return self.id_to_token[token_id]

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab[token]

    def _convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)
