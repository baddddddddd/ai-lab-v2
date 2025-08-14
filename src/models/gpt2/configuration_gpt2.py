from ..base_config import BaseConfig


class GPT2Config(BaseConfig):
    def __init__(
        self,
        vocab_size: int,
        n_ctx: int,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        eos_token_id: int,
    ):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.eos_token_id = eos_token_id

        self.d_head = d_model // n_heads
