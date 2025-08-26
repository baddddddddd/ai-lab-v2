import torch

from .base_kv_cache import BaseKVCache, BaseKVCacheLayer


class StaticKVCacheLayer(BaseKVCacheLayer):
    def __init__(self, max_cache_len: int):
        super().__init__()
        self.max_cache_len = max_cache_len
        self.cache_ptr = 0

    def lazy_initialization(self, key_states: torch.Tensor):
        self.max_batch_size, self.n_heads, _, self.d_head = key_states.shape
        self.device = key_states.device
        self.dtype = key_states.dtype

        self.k_cache = torch.zeros(
            self.max_batch_size,
            self.n_heads,
            self.max_cache_len,
            self.d_head,
            device=self.device,
            dtype=self.dtype,
        )
        self.v_cache = torch.zeros(
            self.max_batch_size,
            self.n_heads,
            self.max_cache_len,
            self.d_head,
            device=self.device,
            dtype=self.dtype,
        )

    def update(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        **cache_kwargs,
    ):
        if self.k_cache is None:
            self.lazy_initialization(k_new)

        seq_len = k_new.size(2)
        self.k_cache[:, :, self.cache_ptr : self.cache_ptr + seq_len, :] = k_new
        self.v_cache[:, :, self.cache_ptr : self.cache_ptr + seq_len, :] = v_new

        self.cache_ptr += seq_len

        return (
            self.k_cache[:, :, : self.cache_ptr + seq_len, :],
            self.v_cache[:, :, : self.cache_ptr + seq_len, :],
        )

    def get_seq_length(self):
        return self.cache_ptr


class StaticKVCache(BaseKVCache):
    def __init__(self, n_layers: int, n_ctx: int):
        layers = [StaticKVCacheLayer(n_ctx) for _ in range(n_layers)]
        super().__init__(layers)
