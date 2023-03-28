import torch
import torch.nn
import numpy as np
import math

from utils.util import add_dimensions


def replace(y, p, n_classes, device):
    boolean_ = torch.bernoulli(p * torch.ones_like(y, device=device)).bool()
    no_class_label = n_classes * torch.ones_like(y, device=device)
    y = torch.where(boolean_, no_class_label, y)
    return y


def dropout_label_for_cfg_training(y, n_noise_samples, n_classes, p, device):
    if y is not None:
        if n_classes is None:
            raise ValueError
        else:
            with torch.no_grad():
                boolean_ = torch.bernoulli(
                    p * torch.ones_like(y, device=device)).bool()
                no_class_label = n_classes * torch.ones_like(y, device=device)
                y = torch.where(boolean_, no_class_label, y)
                y = y.repeat_interleave(n_noise_samples)
                return y
    else:
        return None


class VPSDELoss:
    def __init__(self, beta_min, beta_d, eps_t, n_noise_samples=1, label_unconditioning_prob=.1, n_classes=None, **kwargs):
        self.beta_min = beta_min
        self.beta_d = beta_d
        self.eps_t = eps_t
        self.n_noise_samples = n_noise_samples
        self.label_unconditioning_prob = label_unconditioning_prob
        self.n_classes = n_classes

    def _sigma(self, t):
        return ((.5 * self.beta_d * t ** 2. + self.beta_min * t).exp() - 1.).sqrt()

    def get_loss(self, model, x, y):
        y = dropout_label_for_cfg_training(
            y, self.n_noise_samples, self.n_classes, self.label_unconditioning_prob, x.device)

        t = (1. - self.eps_t) * \
            torch.rand((x.shape[0], self.n_noise_samples),
                       device=x.device) + self.eps_t
        sigma = self._sigma(t)

        sigma = add_dimensions(sigma, len(x.shape) - 1)
        x_repeated = x.unsqueeze(1).repeat_interleave(
            self.n_noise_samples, dim=1)
        x_noisy = x_repeated + sigma * \
            torch.randn_like(x_repeated, device=x.device)

        w = 1. / sigma ** 2.

        pred = model(x_noisy.reshape(-1, *x.shape[1:]), sigma.reshape(-1, *sigma.shape[2:]), y).reshape(
            x.shape[0], self.n_noise_samples, *x.shape[1:])
        loss = w * (pred - x_repeated) ** 2.
        loss = torch.mean(loss.reshape(loss.shape[0], -1), dim=-1)
        return loss


class VESDELoss:
    def __init__(self, sigma_min, sigma_max, n_noise_samples=1, label_unconditioning_prob=.1, n_classes=None, **kwargs):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.n_noise_samples = n_noise_samples
        self.label_unconditioning_prob = label_unconditioning_prob
        self.n_classes = n_classes

    def get_loss(self, model, x, y):
        y = dropout_label_for_cfg_training(
            y, self.n_noise_samples, self.n_classes, self.label_unconditioning_prob, x.device)

        log_sigma = (np.log(self.sigma_max) - np.log(self.sigma_min)) * torch.rand(
            (x.shape[0], self.n_noise_samples), device=x.device) + np.log(self.sigma_min)
        sigma = log_sigma.exp()

        sigma = add_dimensions(sigma, len(x.shape) - 1)
        x_repeated = x.unsqueeze(1).repeat_interleave(
            self.n_noise_samples, dim=1)
        x_noisy = x_repeated + sigma * \
            torch.randn_like(x_repeated, device=x.device)

        w = 1. / sigma ** 2.

        pred = model(x_noisy.reshape(-1, *x.shape[1:]), sigma.reshape(-1, *sigma.shape[2:]), y).reshape(
            x.shape[0], self.n_noise_samples, *x.shape[1:])
        loss = w * (pred - x_repeated) ** 2.
        loss = torch.mean(loss.reshape(loss.shape[0], -1), dim=-1)
        return loss


class VLoss:
    def __init__(self, logsnr_min, logsnr_max, n_noise_samples=1, label_unconditioning_prob=.1, n_classes=None, **kwargs):
        self.logsnr_min = logsnr_min
        self.logsnr_max = logsnr_max
        self.eps_min = self._t_given_logsnr(logsnr_max)
        self.eps_max = self._t_given_logsnr(logsnr_min)
        self.n_noise_samples = n_noise_samples
        self.label_unconditioning_prob = label_unconditioning_prob
        self.n_classes = n_classes

    def _t_given_logsnr(self, logsnr):
        return 2. * np.arccos(1. / np.sqrt(1. + np.exp(-logsnr))) / np.pi

    def _sigma(self, t):
        return (torch.cos(np.pi * t / 2.) ** (-2.) - 1.).sqrt()

    def get_loss(self, model, x, y):
        y = dropout_label_for_cfg_training(
            y, self.n_noise_samples, self.n_classes, self.label_unconditioning_prob, x.device)

        t = (self.eps_max - self.eps_min) * \
            torch.rand((x.shape[0], self.n_noise_samples),
                       device=x.device) + self.eps_min
        sigma = self._sigma(t)

        sigma = add_dimensions(sigma, len(x.shape) - 1)
        x_repeated = x.unsqueeze(1).repeat_interleave(
            self.n_noise_samples, dim=1)
        x_noisy = x_repeated + sigma * \
            torch.randn_like(x_repeated, device=x.device)

        w = (sigma ** 2. + 1.) / sigma ** 2.

        pred = model(x_noisy.reshape(-1, *x.shape[1:]), sigma.reshape(-1, *sigma.shape[2:]), y).reshape(
            x.shape[0], self.n_noise_samples, *x.shape[1:])
        loss = w * (pred - x_repeated) ** 2.
        loss = torch.mean(loss.reshape(loss.shape[0], -1), dim=-1)
        return loss


class EDMLoss:
    def __init__(self, p_mean, p_std, sigma_data=math.sqrt(1. / 3), n_noise_samples=1, label_unconditioning_prob=.1, n_classes=None, **kwargs):
        self.p_mean = p_mean
        self.p_std = p_std
        self.sigma_data = sigma_data
        self.n_noise_samples = n_noise_samples
        self.label_unconditioning_prob = label_unconditioning_prob
        self.n_classes = n_classes

    def get_loss(self, model, x, y):
        y = dropout_label_for_cfg_training(
            y, self.n_noise_samples, self.n_classes, self.label_unconditioning_prob, x.device)

        log_sigma = self.p_mean + self.p_std * \
            torch.randn(
                (x.shape[0], self.n_noise_samples), device=x.device)

        sigma = log_sigma.exp()

        sigma = add_dimensions(sigma, len(x.shape) - 1)
        x_repeated = x.unsqueeze(1).repeat_interleave(
            self.n_noise_samples, dim=1)
        x_noisy = x_repeated + sigma * \
            torch.randn_like(x_repeated, device=x.device)

        w = (sigma ** 2. + self.sigma_data ** 2.) / \
            (sigma * self.sigma_data) ** 2.

        pred = model(x_noisy.reshape(-1, *x.shape[1:]), sigma.reshape(-1, *sigma.shape[2:]), y).reshape(
            x.shape[0], self.n_noise_samples, *x.shape[1:])
        loss = w * (pred - x_repeated) ** 2.
        loss = torch.mean(loss.reshape(loss.shape[0], -1), dim=-1)
        return loss
