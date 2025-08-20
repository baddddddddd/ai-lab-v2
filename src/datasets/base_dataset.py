from typing import Sized

from torch.utils.data import Dataset, IterableDataset


class BaseDataset(Dataset, Sized):
    def __len__(self):
        raise NotImplementedError("__len__() method is not implemented")

    def __getitem__(self, idx: int):
        raise NotImplementedError("__getitem__() method is not implemented")

    def get_dataloader(self, batch_size: int, shuffle: bool = False):
        raise NotImplementedError("get_dataloader() method is not implemented")


class BaseStreamingDataset(IterableDataset):
    def __iter__(self):
        raise NotImplementedError("__iter__() method is not implemented")

    def get_dataloader(self, batch_size: int):
        raise NotImplementedError("get_dataloader() method is not implemented")
