import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import BaseStreamer, top_p_sample
from ..base_model import BaseModel
from ..model_output import CausalLmOutput
from .configuration_llama import LlamaConfig


def precompute_cos_sin_tables(n_ctx: int, d_head):
    positions = torch.arange(n_ctx, dtype=torch.float32)
    dims = torch.arange(d_head, dtype=torch.float32) // 2
    theta = torch.pow(10000, -2 * dims / d_head)
    angles = positions[:, None] * theta[None, :]

    return angles.cos(), angles.sin()


def rotate_half(x: torch.Tensor):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    return x * cos + rotate_half(x) * sin


class LlamaKVCache:
    def __init__(self, config: LlamaConfig, device: torch.device, batch_size: int = 1):
        self.k_cache = torch.zeros(
            config.n_layers,
            batch_size,
            config.n_heads,
            config.n_ctx,
            config.d_head,
            device=device,
        )
        self.v_cache = torch.zeros(
            config.n_layers,
            batch_size,
            config.n_heads,
            config.n_ctx,
            config.d_head,
            device=device,
        )

    def update(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        layer_idx: int,
        cache_position: int,
    ):
        seq_len = k_new.size(2)
        self.k_cache[layer_idx, :, :, cache_position : cache_position + seq_len, :] = (
            k_new
        )
        self.v_cache[layer_idx, :, :, cache_position : cache_position + seq_len, :] = (
            v_new
        )

        return (
            self.k_cache[layer_idx, :, :, : cache_position + seq_len, :],
            self.v_cache[layer_idx, :, :, : cache_position + seq_len, :],
        )


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
        past_key_values: LlamaKVCache | None = None,
        cache_position: int = 0,
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
            k, v = past_key_values.update(k, v, self.layer_idx, cache_position)

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
        past_key_values: LlamaKVCache | None = None,
        cache_position: int = 0,
    ):
        attn_output = self.attn(
            self.attn_norm(x),
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
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
            [
                LlamaBlock(config, layer_idx=layer_idx)
                for layer_idx in range(config.n_layers)
            ]
        )

        self.final_norm = nn.RMSNorm(config.d_model, eps=1e-6)

        cos, sin = precompute_cos_sin_tables(config.n_ctx, config.d_head)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

        self.apply(self._init_weights)

        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        past_key_values: LlamaKVCache | None = None,
        cache_position: int = 0,
        use_cache: bool = False,
    ) -> CausalLmOutput:
        batch_size, seq_len = input_ids.size()

        if use_cache and past_key_values is None:
            past_key_values = LlamaKVCache(
                self.config,
                batch_size=batch_size,
                device=input_ids.device,
            )

        token_embeds = self.token_embedding(input_ids)
        hidden_state = token_embeds

        cos = self.cos_cached[cache_position : cache_position + seq_len]
        sin = self.sin_cached[cache_position : cache_position + seq_len]
        position_embeddings = (cos, sin)

        for block in self.decoder_stack:
            hidden_state = block(
                hidden_state,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                cache_position=cache_position,
            )

        hidden_state = self.final_norm(hidden_state)

        logits = hidden_state @ self.token_embedding.weight.T

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

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.9,
        streamer: BaseStreamer | None = None,
    ):
        generated = input_ids.clone().tolist()
        if streamer is not None:
            streamer.put(input_ids)

        model_input = torch.tensor(
            [generated],
            dtype=torch.long,
            device=input_ids.device,
        )
        past_key_values = None
        cache_position = 0
        for step in range(max_new_tokens):
            output = self.forward(
                model_input,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=True,
            )
            logits = output.logits[0, -1]
            past_key_values = output.past_key_values
            cache_position = len(generated)

            if temperature > 0.0:
                logits /= temperature
                if top_p > 0.0:
                    next_token = top_p_sample(logits, p=top_p).item()
                else:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = torch.argmax(logits).item()

            generated.append(next_token)
            model_input = torch.tensor(
                [[next_token]], dtype=torch.long, device=input_ids.device
            )

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
