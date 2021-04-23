import torch as th
from numpy import angle
from math import pi
import numpy as np


def hann_window(L):
    time = th.arange(L)
    window = th.sin(pi * time /(L - 1)) ** 2
    return window


def from_time_to_frequency(tensor, window):
    batch = True if len(tensor.shape) == 2 else False
    window = window(tensor.shape[-1])
    if not batch:
        tensor = tensor.unsqueeze(0)
    window = th.cat(tensor.shape[0] * [window.unsqueeze(0)]).to(tensor.device)
    freq_domain_tensor = th.rfft(window * tensor, signal_ndim=1, onesided=True, normalized=True).permute(0, 2, 1)
    if not batch:
        freq_domain_tensor = freq_domain_tensor.squeeze(0)
    return freq_domain_tensor


def from_fft_evaluate_abs(tensor):
    batch = True if len(tensor.shape) == 3 else False
    if not batch:
        tensor = tensor.unsqueeze(0)
    fft_abs = th.norm(tensor.permute(0, 2, 1), dim=2)
    if not batch:
        fft_abs = fft_abs.squeeze(0)
    return fft_abs


def from_fft_evaluate_angle(tensor):
    batch = True if len(tensor.shape) == 3 else False
    if not batch:
        tensor = tensor.unsqueeze(0)
    fft_angle = angle(tensor[:, 0, :].numpy() + 1j * tensor[:, 1, :].numpy())
    if not batch:
        fft_angle = fft_angle.squeeze(0)
    return th.tensor(fft_angle)


def get_fourier_abs(tensor, window):
    dft = from_time_to_frequency(tensor, window)
    return from_fft_evaluate_abs(dft)