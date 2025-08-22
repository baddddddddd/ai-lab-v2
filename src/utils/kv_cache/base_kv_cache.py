import torch


class BaseKVCacheLayer:
    def __init__(self):
        self.k_cache = None
        self.v_cache = None

    def update(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        **cache_kwargs,
    ):
        raise NotImplementedError("update() method is not implemented")

    def get_seq_length(self):
        raise NotImplementedError("get_seq_length() method is not implemented")


class BaseKVCache:
    def __init__(self, layers: list[BaseKVCacheLayer] | None = None):
        self.layers = layers if layers else []

    def update(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        layer_idx: int,
        **cache_kwargs,
    ):
        k, v = self.layers[layer_idx].update(k_new, v_new, **cache_kwargs)
        return k, v

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.layers):
            return 0

        return self.layers[layer_idx].get_seq_length()
