import numpy as np
import copy
from torch.utils.data import Dataset


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
        x, y = np.load(self.x[i], allow_pickle = True), self.y[i]
        if self.process is not None:
            x, y = self.process(x, y)
        if len(x.shape) == 1:
            return x[None, :], y
        else:
            return x, y


class ContainerDemo(BaseContainer):
    def __init__(self, x, y, f_val=0.2, f_test=0, split="all", blend=True, dilate=1):
        super().__init__(x, y, f_val=f_val, f_test=f_test, split=split)

        fix_noise = split in ["val", "tst"]

        np.random.seed(0)
        self.rand = np.random.normal(0, 1, size=self.L) if fix_noise else None
        self.blend = blend
        self.dim_alpha = 4

    def process(self, x, y, baseSN=None, f_s=None):
        # if np.random.uniform() > 0.5:
        # x, y = self.flip_time(x, y)
        # total_flux, params = self.add_blend(x, y)
        # m_obs = self.noisify(x)
        # m_obs, params = self.sample_t0(m_obs, params)

        return x[None, :], y

    def noisify(self, total_flux):
        baseSN = np.random.uniform(23, 230)
        flux_err = total_flux**0.5 / baseSN

        # noise is fixed during validation
        rand = (
            self.rand
            if self.rand is not None
            else np.random.normal(0, 1, size=flux_err.shape[0])
        )
        m_obs = total_flux + rand * flux_err
        return m_obs

    def add_blend(self, x, y):
        f_s = 10 ** np.random.uniform(-1.3, 0) if self.blend else 1
        x = (1 / f_s - 1) + x
        y = np.append(y, f_s)
        return x, y
