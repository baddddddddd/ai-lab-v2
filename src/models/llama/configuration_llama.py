from ..base_config import BaseConfig


class LlamaConfig(BaseConfig):
    def __init__(
        self,
        vocab_size: int,
        n_ctx: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
    ):
        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.d_ff = 8 * d_model // 3
        self.d_head = d_model // n_heads
