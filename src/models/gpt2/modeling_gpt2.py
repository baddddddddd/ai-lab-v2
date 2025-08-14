import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration_gpt2 import GPT2Config
from ..base_model import BaseModel


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


class NewGELU(nn.Module):
    def forward(self, x):
        return (
            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.0356774 * x**3)))
        )


class MultiHeadCausalAttention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()

        self.config = config

        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

    def forward(self, x: torch.Tensor):
        # x: (batch_size, n_ctx, d_model)
        batch_size, n_ctx, _ = x.size()

        qkv = self.qkv_proj(x)  # (batch_size, n_ctx, 3 * d_model)
        qkv = qkv.reshape(batch_size, n_ctx, 3, self.config.n_heads, self.config.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, n_heads, n_ctx, d_head)
        q, k, v = qkv

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.config.dropout if self.training else 0.0,
        )
        output = output.transpose(-2, -3).reshape(
            batch_size, n_ctx, self.config.d_model
        )
        output = self.out_proj(output)
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            NewGELU(),
            nn.Linear(config.d_ff, config.d_model),
        )

        self.config = config

    def forward(self, x: torch.Tensor):
        return self.ff(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()

        self.pre_attn_norm = nn.LayerNorm(config.d_model, eps=1e-5)
        self.attn = MultiHeadCausalAttention(config)
        self.post_attn_dropout = nn.Dropout(config.dropout)

        self.pre_ff_norm = nn.LayerNorm(config.d_model, eps=1e-5)
        self.ff = PositionWiseFeedForward(config)
        self.post_ff_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor):
        attn_output = self.attn(self.pre_attn_norm(x))
        x = x + self.post_attn_dropout(attn_output)

        ff_output = self.ff(self.pre_ff_norm(x))
        x = x + self.post_ff_dropout(ff_output)

        return x


class Embedding(nn.Module):
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

    def forward(self, input_ids: torch.Tensor):
        # x: (batch_size, seq_len)
        n_ctx = input_ids.size(-1)

        positions = torch.arange(n_ctx, device=input_ids.device).unsqueeze(0)

        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(positions)

        embeddings = token_embeds + position_embeds
        embeddings = self.dropout(embeddings)

        return embeddings


class GPT2Model(BaseModel):
    config_class = GPT2Config

    def __init__(self, config: GPT2Config):
        super().__init__(config)

        self.config = config
        self.embedding = Embedding(config)
        self.transformer_stack = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.final_norm = nn.LayerNorm(config.d_model, eps=1e-5)

        self.apply(self._init_weights)
        self._scale_residual_weights()

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

    def forward(self, input_ids: torch.Tensor):
        hidden = self.embedding(input_ids)

        for block in self.transformer_stack:
            hidden = block(hidden)

        hidden = self.final_norm(hidden)

        logits = hidden @ self.embedding.token_embedding.weight.T
        return logits
