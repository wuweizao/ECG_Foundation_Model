from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, filtfilt, iirnotch, medfilt
from torch.utils.data import Dataset

from .labels import CLASSES

LEADS = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
PREPROCESS_VERSION = 'founder-filter-global-z-v1'


def preprocess(signal, names, fs):
    if fs != 500 or signal.shape != (5000, 12):
        raise ValueError(f'Expected native 500 Hz, 10 s, 12 leads; got {fs}, {signal.shape}')
    names = [s.upper() for s in names]
    x = signal[:, [names.index(s) for s in LEADS]].T.astype(np.float64)
    if not np.isfinite(x).all():
        raise ValueError('Non-finite waveform; explicit audit required')
    b, a = iirnotch(50, 30, fs)
    x = filtfilt(b, a, x, axis=-1)
    b, a = butter(4, [0.67, 40], btype='bandpass', fs=fs)
    x = filtfilt(b, a, x, axis=-1)
    x = x - np.stack([medfilt(lead, kernel_size=201) for lead in x])
    return ((x - x.mean()) / (x.std() + 1e-8)).astype(np.float32)


class ECGDataset(Dataset):
    def __init__(self, frame, cache):
        self.frame = frame.reset_index(drop=True)
        self.cache = Path(cache)
        self.labels = self.frame[CLASSES].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        x = np.load(self.cache / f'{int(self.frame.iloc[i].ecg_id)}.npy')
        return torch.from_numpy(x), torch.from_numpy(self.labels[i])
