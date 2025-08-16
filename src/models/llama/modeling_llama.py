import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import BaseStreamer, top_p_sample
from ..base_model import BaseModel
from .configuration_llama import LlamaConfig


def rotate_half(x: torch.Tensor):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    return x * cos + rotate_half(x) * sin


class LlamaAttention(nn.Module):
    """
    Self-attention with RoPE
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()

        self.config = config

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor, position_embeddings: torch.Tensor):
        # (batch_size, n_ctx, d_model)
        batch_size, n_ctx, _ = x.size()
        proj_reshape = (batch_size, n_ctx, self.config.n_heads, self.config.d_head)

        q = self.q_proj(x).view(proj_reshape).transpose(1, 2)
        k = self.k_proj(x).view(proj_reshape).transpose(1, 2)
        v = self.v_proj(x).view(proj_reshape).transpose(1, 2)

        cos, sin = position_embeddings
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        values = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        values = (
            values.transpose(1, 2)
            .contiguous()
            .reshape(batch_size, n_ctx, self.config.d_model)
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
    def __init__(self, config: LlamaConfig):
        super().__init__()

        self.attn_norm = nn.RMSNorm(config.d_model, eps=1e-6)
        self.attn = LlamaAttention(config)
        self.mlp_norm = nn.RMSNorm(config.d_model, eps=1e-6)
        self.mlp = LlamaMLP(config)

    def forward(self, x: torch.Tensor, position_embeddings: torch.Tensor):
        attn_output = self.attn(self.attn_norm(x), position_embeddings)
        x = x + attn_output

        mlp_output = self.mlp(self.mlp_norm(x))
        x = x + mlp_output

        return x


class LlamaModel(BaseModel):
    config_class = LlamaConfig

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
        )

        self.decoder_stack = nn.ModuleList(
            [LlamaBlock(config) for _ in range(config.n_layers)]
        )

        self.final_norm = nn.RMSNorm(config.d_model, eps=1e-6)

        positions = torch.arange(config.n_ctx, dtype=torch.float32)
        dims = torch.arange(config.d_head, dtype=torch.float32) // 2
        theta = torch.pow(10000, -2 * dims / config.d_head)
        angles = positions[:, None] * theta[None, :]

        self.register_buffer("cos_cached", angles.cos(), persistent=True)
        self.register_buffer("sin_cached", angles.sin(), persistent=True)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor):
        seq_len = input_ids.size(1)

        token_embeds = self.token_embedding(input_ids)
        hidden_state = token_embeds

        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        position_embeddings = (cos, sin)

        for block in self.decoder_stack:
            hidden_state = block(hidden_state, position_embeddings)

        hidden_state = self.final_norm(hidden_state)

        logits = hidden_state @ self.token_embedding.weight.T
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.9,
        streamer: BaseStreamer | None = None,
    ):
        from collections import deque

        generated = deque(input_ids.clone().tolist(), maxlen=self.config.n_ctx)
        if streamer is not None:
            streamer.put(input_ids)

        for _ in range(max_new_tokens):
            model_input = torch.tensor(
                [generated], dtype=torch.long, device=input_ids.device
            )
            logits = self.forward(model_input)
            logits = logits[0, -1]

            if temperature > 0.0:
                logits /= temperature
                if top_p > 0.0:
                    next_token = top_p_sample(logits, p=top_p)
                else:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = torch.argmax(logits).item()

            generated.append(next_token)

            if streamer is not None:
                streamer.put(torch.tensor([next_token], device=input_ids.device))

            if (
                self.config.eos_token_id is not None
                and next_token == self.config.eos_token_id
            ):
                break

        if streamer is not None:
            streamer.end()

        return torch.tensor(generated, device=input_ids.device)
