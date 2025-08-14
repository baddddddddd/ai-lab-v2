import os
import pathlib
from typing import Self

import torch
import torch.nn as nn

from .base_config import BaseConfig


class BaseModel(nn.Module):
    SAVE_FILENAME = "pytorch_model.bin"
    config_class = BaseConfig

    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config

    def size(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_pretrained(
        cls, pretrained_model_path: str | os.PathLike, device_map: str = "auto"
    ) -> Self:
        if device_map == "auto":
            device_map = "cuda" if torch.cuda.is_available() else "cpu"

        model_folder = pathlib.Path(pretrained_model_path).resolve()
        model_file = model_folder / cls.SAVE_FILENAME

        config = cls.config_class.from_pretrained(model_folder)
        model = cls(config)

        model_state_dict = torch.load(model_file, map_location=device_map)
        model.load_state_dict(model_state_dict)
        model.to(device_map)

        return model

    def save_pretrained(self, save_directory: str | os.PathLike):
        save_folder = pathlib.Path(save_directory).resolve()
        model_path = save_folder / BaseModel.SAVE_FILENAME

        if os.path.exists(model_path):
            raise ValueError(f"{model_path} file already exists")

        os.makedirs(save_folder, exist_ok=True)

        torch.save(self.state_dict(), model_path)

        self.config.save_pretrained(save_folder)
