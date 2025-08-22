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
