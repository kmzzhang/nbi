import copy

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import Subset


class Data:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.N = len(x)  # number of datapoints
        self.y = self.y.astype("float32")


class BaseContainer(Dataset):
    def __init__(self, x, y, f_val=0.2, f_test=0, split="all", process=None):
        # create data partitions
        N = len(x)
        p_train = 1 - f_val - f_test
        p_val = 1 - f_test

        self.trn = Data(x[: int(N * p_train)], y[: int(N * p_train)])
        self.val = Data(
            x[int(N * p_train) : int(N * p_val)], y[int(N * p_train) : int(N * p_val)]
        )
        self.tst = Data(x[int(N * p_val) :], y[int(N * p_val) :])
        self.all = Data(x, y)
        self.set_split(split)
        self.process = process

    def set_split(self, split="all"):
        data = getattr(self, split)
        self.split = split
        self.x = data.x
        self.y = data.y

    def get_splits(self):
        train = copy.copy(self)
        val = copy.copy(self)
        test = copy.copy(self)
        train.set_split("trn")
        val.set_split("val")
        test.set_split("tst")
        return train, val, test

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i, **kwargs):
        if isinstance(self.x[i], np.str_):
            x, y = np.load(self.x[i], allow_pickle=True), self.y[i]
        else:
            x, y = self.x[i], self.y[i]
        if self.process is not None:
            x, y = self.process(x, y)

        return np.atleast_2d(x), y

class ProcessedTorchDataset(Dataset):
    """
    Wrap a torch Dataset and apply `process` on-the-fly in __getitem__.

    Supports datasets that return:
      - x
      - (x, y)
    and supports process signatures:
      - process(x, y) -> (x2, y2)
      - process(x)    -> x2
    """
    def __init__(self, dataset, process=None):
        self.dataset = dataset
        self.process = process

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        out = self.dataset[idx]

        # Normalize to (x, y)
        if isinstance(out, (tuple, list)) and len(out) == 2:
            x, y = out
        else:
            x, y = out, None

        if self.process is None:
            return out

        # Apply process
        try:
            outp = self.process(x, y)
        except TypeError:
            outp = self.process(x)

        return outp

class DatasetContainer:
    """
    Container wrapper for torch Datasets so NBI can reuse _init_loader().
    Provides the same get_splits() API as BaseContainer, and supports `process`
    applied in __getitem__ (like BaseContainer).
    """
    def __init__(self, dataset, f_val=0.1, f_test=0.0, seed=0, process=None):
        self.dataset = dataset
        self.f_val = float(f_val)
        self.f_test = float(f_test)
        self.seed = int(seed)
        self.process = process

        # Wrap so process is applied for any subset indexing
        if self.process is not None:
            self.dataset = ProcessedTorchDataset(self.dataset, self.process)

    def get_splits(self):
        n = len(self.dataset)
        rng = np.random.default_rng(self.seed)
        idx = np.arange(n)
        rng.shuffle(idx)

        n_test = int(self.f_test * n)
        n_val  = int(self.f_val * n)

        test_idx  = idx[:n_test]
        val_idx   = idx[n_test:n_test + n_val]
        train_idx = idx[n_test + n_val:]

        return (
            Subset(self.dataset, train_idx),
            Subset(self.dataset, val_idx),
            Subset(self.dataset, test_idx),
        )