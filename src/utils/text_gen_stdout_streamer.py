from .base_streamer import BaseStreamer
from ..tokenizers import BaseTokenizer


class TextGenStdoutStreamer(BaseStreamer):
    def __init__(self, tokenizer: BaseTokenizer):
        self.tokenizer = tokenizer

    def put(self, token_ids):
        decoded = self.tokenizer.decode(token_ids)
        print(decoded, end="", flush=True)

    def end(self):
        print()
