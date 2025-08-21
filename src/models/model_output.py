from dataclasses import dataclass, fields

import torch


class ModelOutput(dict):
    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            self[f.name] = value
            setattr(self, f.name, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
        super().__setitem__(key, value)

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()})"


@dataclass
class CausalLmOutput(ModelOutput):
    logits: torch.FloatTensor | None = None
    loss: torch.FloatTensor | None = None
    past_key_values: torch.FloatTensor | None = None
