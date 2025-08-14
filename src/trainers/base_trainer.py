import os


class BaseTrainer:
    def train(self, resume_from_checkpoint: os.PathLike[str] | bool = False):
        raise NotImplementedError("train() method is not implemented")
