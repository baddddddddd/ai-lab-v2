from __future__ import annotations

import json
import os
import pathlib
from typing import Self


class BaseConfig:
    SAVE_FILENAME = "config.json"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str | os.PathLike,
    ) -> Self:
        model_path = pathlib.Path(pretrained_model_path).resolve()
        config_filepath = model_path / BaseConfig.SAVE_FILENAME

        return cls.from_json_file(config_filepath)

    @classmethod
    def from_json_file(cls, json_file: str | os.PathLike) -> Self:
        with open(json_file, "r") as f:
            content = f.read()
            config_dict = json.loads(content)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict) -> Self:
        config = cls.__new__(cls)
        config.__dict__.update(config_dict)
        return config

    def save_pretrained(self, save_directory):
        save_folder = pathlib.Path(save_directory).resolve()
        os.makedirs(save_folder, exist_ok=True)

        json_file_path = save_folder / BaseConfig.SAVE_FILENAME
        self.to_json_file(json_file_path)

    def to_json_file(self, json_file_path: str | os.PathLike):
        if os.path.exists(json_file_path):
            raise ValueError(f"{json_file_path} file already exists")

        with open(json_file_path, "w") as f:
            json_string = self.to_json_string()
            f.write(json_string)

    def to_json_string(self) -> str:
        config_dict = self.to_dict()
        json_string = json.dumps(config_dict)
        return json_string

    def to_dict(self):
        return self.__dict__
