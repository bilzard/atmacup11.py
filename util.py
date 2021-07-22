import random
import os

import matplotlib.pyplot as plt
import numpy as np
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


class SimpleLogger():
    def __init__(self):
        self._log = []

    def add(self, value):
        self._log.append(value)

    def plot(self, title, ma_length=20, file_name=None):
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        ax.plot(self._log, label='loss')
        ax.plot(moving_average(self._log, ma_length), label=f'loss_ma({ma_length})')
        ax.set_yscale('log')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend()
        if file_name:
            plt.savefig(file_name)
        plt.show()


def seed_everything(seed: int=2929, deterministic: bool=False):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic

