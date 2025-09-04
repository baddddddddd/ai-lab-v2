from ..base_config import BaseConfig


class LlamaConfig(BaseConfig):
    def __init__(
        self,
        vocab_size: int,
        n_ctx: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        bos_token_id: int,
        eos_token_id: int,
        rope_theta: float = 10000.0,
    ):
        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.rope_theta = rope_theta

        self.d_ff = 8 * d_model // 3
        self.d_head = d_model // n_heads
