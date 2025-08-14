from typing import Sized

from torch.utils.data import Dataset


class BaseDataset(Dataset, Sized):
    pass
