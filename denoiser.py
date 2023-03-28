import math
import numpy as np
import torch
import torch.nn as nn


class EDMDenoiser(nn.Module):
    def __init__(self,
                 model,
                 sigma_data=math.sqrt(1. / 3)):

        super().__init__()

        self.sigma_data = sigma_data
        self.model = model

    def forward(self, x, sigma, y=None):
        c_skip = self.sigma_data ** 2. / \
            (sigma ** 2. + self.sigma_data ** 2.)
        c_out = sigma * self.sigma_data / \
            torch.sqrt(self.sigma_data ** 2. + sigma ** 2.)
        c_in = 1. / torch.sqrt(self.sigma_data ** 2. + sigma ** 2.)
        c_noise = .25 * torch.log(sigma)

        out = self.model(c_in * x, c_noise.reshape(-1), y)
        x_denoised = c_skip * x + c_out * out
        return x_denoised


class VDenoiser(nn.Module):
    def __init__(self,
                 model):

        super().__init__()

        self.model = model

    def _sigma_inv(self, sigma):
        return 2. * torch.arccos(1. / (1. + sigma ** 2.).sqrt()) / np.pi

    def forward(self, x, sigma, y=None):
        c_skip = 1. / (sigma ** 2. + 1.)
        c_out = sigma / torch.sqrt(1. + sigma ** 2.)
        c_in = 1. / torch.sqrt(1. + sigma ** 2.)
        c_noise = self._sigma_inv(sigma)

        out = self.model(c_in * x, c_noise.reshape(-1), y)
        x_denoised = c_skip * x + c_out * out
        return x_denoised


class VESDEDenoiser(nn.Module):
    def __init__(self,
                 model):

        super().__init__()

        self.model = model

    def forward(self, x, sigma, y=None, context=None):
        c_skip = 1.
        c_out = sigma
        c_in = 1.
        c_noise = torch.log(sigma / 2.)

        out = self.model(c_in * x, c_noise.reshape(-1), y, context)
        x_denoised = c_skip * x + c_out * out
        return x_denoised


class VPSDEDenoiser(nn.Module):
    def __init__(self,
                 beta_min,
                 beta_d,
                 M,
                 model):

        super().__init__()

        self.model = model
        self.M = M
        self.beta_min = beta_min
        self.beta_d = beta_d

    def _sigma_inv(self, sigma):
        beta_ratio = self.beta_min / self.beta_d
        return -beta_ratio + (beta_ratio ** 2. + 2. * torch.log(sigma ** 2. + 1.) / self.beta_d).sqrt()

    def forward(self, x, sigma, y=None, context=None):
        c_skip = 1.
        c_out = -sigma
        c_in = 1. / torch.sqrt(sigma ** 2. + 1.)
        c_noise = self.M * self._sigma_inv(sigma)

        out = self.model(c_in * x, c_noise.reshape(-1), y, context)
        x_denoised = c_skip * x + c_out * out
        return x_denoised
