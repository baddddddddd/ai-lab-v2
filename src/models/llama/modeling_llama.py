import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import BaseKVCache, StaticKVCache, CausalLmGenerationMixin
from ..base_model import BaseModel
from ..model_output import CausalLmOutput
from .configuration_llama import LlamaConfig


# static RoPE
# def precompute_cos_sin_tables(n_ctx: int, d_head: int):
#     positions = torch.arange(n_ctx, dtype=torch.float32)
#     dims = torch.arange(d_head, dtype=torch.float32) // 2
#     theta = torch.pow(10000, -2 * dims / d_head)
#     angles = positions[:, None] * theta[None, :]

#     return angles.cos(), angles.sin()


# ntk-aware scaling
def precompute_cos_sin_tables(
    n_ctx: int,
    d_head: int,
    base: float = 10000.0,
    factor: float = 1.0,
):
    positions = torch.arange(n_ctx, dtype=torch.float32)
    dims = torch.arange(d_head, dtype=torch.float32) // 2

    ntk_factor = factor ** (d_head / (d_head - 2))
    adjusted_base = base * ntk_factor

    theta = torch.pow(adjusted_base, -2 * dims / d_head)
    angles = positions[:, None] * theta[None, :]

    return angles.cos(), angles.sin()


def rotate_half(x: torch.Tensor):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    return x * cos + rotate_half(x) * sin


class LlamaAttention(nn.Module):
    """
    Self-attention with RoPE
    """

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: torch.Tensor,
        past_key_values: BaseKVCache | None = None,
    ):
        # (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.size()
        proj_reshape = (batch_size, seq_len, self.config.n_heads, self.config.d_head)

        q = self.q_proj(x).view(proj_reshape).transpose(1, 2)
        k = self.k_proj(x).view(proj_reshape).transpose(1, 2)
        v = self.v_proj(x).view(proj_reshape).transpose(1, 2)

        cos, sin = position_embeddings
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        values = F.scaled_dot_product_attention(q, k, v, is_causal=seq_len > 1)
        values = (
            values.transpose(1, 2)
            .contiguous()
            .reshape(batch_size, seq_len, self.config.d_model)
        )

        out = self.o_proj(values)
        return out


class LlamaMLP(nn.Module):
    """
    Uses SwiGLU for FFN
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()

        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.silu = nn.SiLU()
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor):
        out = self.down_proj(self.silu(self.gate_proj(x)) * self.up_proj(x))
        return out


class LlamaBlock(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()

        self.attn_norm = nn.RMSNorm(config.d_model, eps=1e-6)
        self.attn = LlamaAttention(config, layer_idx=layer_idx)
        self.mlp_norm = nn.RMSNorm(config.d_model, eps=1e-6)
        self.mlp = LlamaMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: torch.Tensor,
        past_key_values: BaseKVCache | None = None,
    ):
        attn_output = self.attn(
            self.attn_norm(x),
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
        )
        x = x + attn_output

        mlp_output = self.mlp(self.mlp_norm(x))
        x = x + mlp_output

        return x


class LlamaModel(BaseModel, CausalLmGenerationMixin):
    config_class = LlamaConfig

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.config: LlamaConfig

        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
        )

        self.decoder_stack = nn.ModuleList(
            [
                LlamaBlock(config, layer_idx=layer_idx)
                for layer_idx in range(config.n_layers)
            ]
        )

        self.final_norm = nn.RMSNorm(config.d_model, eps=1e-6)

        self.precompute_rope_cache()

        self.apply(self._init_weights)

        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def precompute_rope_cache(self, factor: float = 1.0):
        cos, sin = precompute_cos_sin_tables(
            n_ctx=self.config.n_ctx,
            d_head=self.config.d_head,
            base=self.config.rope_theta,
            factor=factor,
        )
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def extend_rope(self, factor: float):
        self.config.n_ctx = int(self.config.n_ctx * factor)
        self.precompute_rope_cache(factor)

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor | None = None,
        past_key_values: BaseKVCache | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLmOutput:
        batch_size, seq_len = input_ids.size()

        if use_cache and past_key_values is None:
            past_key_values = StaticKVCache(
                n_layers=self.config.n_layers,
                n_ctx=self.config.n_ctx,
            )

        if position_ids is None:
            cached_tokens = past_key_values.get_seq_length() if past_key_values else 0
            position_ids = torch.arange(
                cached_tokens, cached_tokens + seq_len, device=input_ids.device
            ).unsqueeze(0)

        token_embeds = self.token_embedding(input_ids)
        hidden_state = token_embeds

        cos = self.cos_cached[position_ids]
        sin = self.sin_cached[position_ids]
        position_embeddings = (cos, sin)

        for block in self.decoder_stack:
            hidden_state = block(
                hidden_state,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
            )

        hidden_state = self.final_norm(hidden_state)

        logits = hidden_state @ self.token_embedding.weight.T

        loss = None
        if labels is not None:
            x = logits.reshape(-1, logits.size(-1))
            y = labels.reshape(-1)

            loss = self.loss_func(x, y)

        return CausalLmOutput(
            logits=logits,
            loss=loss,
            past_key_values=past_key_values,
        )
