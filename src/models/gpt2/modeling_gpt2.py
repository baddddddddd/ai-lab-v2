import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import BaseKVCache, StaticKVCache, CausalLmGenerationMixin
from ..base_model import BaseModel
from ..model_output import CausalLmOutput
from .configuration_gpt2 import GPT2Config


def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor
):
    # q, k, v: (batch_size, n_heads, n_ctx, d_k)
    d_k = q.size(-1)
    n_ctx = q.size(-2)
    scale_factor = 1 / math.sqrt(d_k)

    attn_scores = (q @ k.transpose(-1, -2)) * scale_factor
    attn_scores = attn_scores.masked_fill(attn_mask.logical_not(), float("-inf"))

    attn_weights = F.softmax(attn_scores, dim=-1)
    values = attn_weights @ v

    return values


class GPT2Attention(nn.Module):
    def __init__(self, config: GPT2Config, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

    def forward(self, x: torch.Tensor, past_key_values: BaseKVCache | None = None):
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.size()

        qkv = self.qkv_proj(x)  # (batch_size, n_ctx, 3 * d_model)
        qkv = qkv.reshape(
            batch_size, seq_len, 3, self.config.n_heads, self.config.d_head
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, n_heads, n_ctx, d_head)
        q, k, v = qkv

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=seq_len > 1,
            dropout_p=self.config.dropout if self.training else 0.0,
        )
        output = output.transpose(-2, -3).reshape(
            batch_size, seq_len, self.config.d_model
        )
        output = self.out_proj(output)
        return output


class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.d_ff, config.d_model),
        )

        self.config = config

    def forward(self, x: torch.Tensor):
        return self.ff(x)


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config, layer_idx: int):
        super().__init__()

        self.pre_attn_norm = nn.LayerNorm(config.d_model, eps=1e-5)
        self.attn = GPT2Attention(config, layer_idx=layer_idx)
        self.post_attn_dropout = nn.Dropout(config.dropout)

        self.pre_ff_norm = nn.LayerNorm(config.d_model, eps=1e-5)
        self.ff = GPT2MLP(config)
        self.post_ff_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, past_key_values: BaseKVCache | None = None):
        attn_output = self.attn(self.pre_attn_norm(x), past_key_values=past_key_values)
        x = x + self.post_attn_dropout(attn_output)

        ff_output = self.ff(self.pre_ff_norm(x))
        x = x + self.post_ff_dropout(ff_output)

        return x


class GPT2Embedding(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()

        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=config.n_ctx,
            embedding_dim=config.d_model,
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: torch.Tensor, start_pos: int = 0):
        # input_ids: (batch_size, seq_len)
        seq_len = input_ids.size(-1)

        positions = torch.arange(
            start_pos, start_pos + seq_len, device=input_ids.device
        ).unsqueeze(0)

        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(positions)

        embeddings = token_embeds + position_embeds
        embeddings = self.dropout(embeddings)

        return embeddings


class GPT2Model(BaseModel, CausalLmGenerationMixin):
    config_class = GPT2Config

    def __init__(self, config: GPT2Config):
        super().__init__(config)

        self.config = config
        self.embedding = GPT2Embedding(config)
        self.transformer_stack = nn.ModuleList(
            [
                GPT2Block(config, layer_idx=layer_idx)
                for layer_idx in range(config.n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(config.d_model, eps=1e-5)

        self.apply(self._init_weights)
        self._scale_residual_weights()

        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _scale_residual_weights(self):
        scale = 1 / math.sqrt(self.config.n_layers)
        for block in self.transformer_stack:
            block.attn.out_proj.weight.data *= scale
            block.ff.ff[2].weight.data *= scale

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        past_key_values: BaseKVCache | None = None,
        start_pos: int = 0,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLmOutput:
        if use_cache and past_key_values is None:
            past_key_values = StaticKVCache(
                n_layers=self.config.n_layers,
                n_ctx=self.config.n_ctx,
            )

        hidden = self.embedding(input_ids, start_pos=start_pos)

        for block in self.transformer_stack:
            hidden = block(hidden, past_key_values=past_key_values)

        hidden = self.final_norm(hidden)

        logits = hidden @ self.embedding.token_embedding.weight.T

        loss = None
        if labels is not None:
            x = logits.view(-1, logits.size(-1))
            y = labels.view(-1)

            loss = self.loss_func(x, y)

        return CausalLmOutput(
            logits=logits,
            loss=loss,
            past_key_values=past_key_values,
        )
