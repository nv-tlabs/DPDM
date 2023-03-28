import torch

from utils.util import add_dimensions


def guidance_wrapper(denoiser, guid_scale,):
    def guidance_denoiser(x, t, y):
        if guid_scale > 0:
            no_class_label = denoiser.module.model.label_dim * torch.ones_like(y, device=x.device)
            return (1. + guid_scale) * denoiser(x, t, y) - guid_scale * denoiser(x, t, no_class_label)
        else:
            return denoiser(x, t, y)

    return guidance_denoiser


def edm_sampler(x, y, denoiser, num_steps, tmin, tmax, rho, guid_scale=None, s_noise=1, s_churn=0, s_min=0, s_max=float('inf'), churn_limit=1, **kwargs):
    t_steps = torch.linspace(tmax ** (1. / rho), tmin **
                             (1. / rho), steps=num_steps, device=x.device) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
    x_cur = x * t_steps[0]

    if guid_scale is not None:
        denoiser = guidance_wrapper(denoiser, guid_scale)

    for i, (t0, t1) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        gamma = min(s_churn / num_steps,
                    churn_limit) if s_min <= t0 <= s_max else 0
        t_hat = t0 + gamma * t0

        if gamma > 0:
            x_hat = x_cur + (s_noise * (t_hat ** 2 - t0 ** 2).sqrt()
                             ) * torch.randn_like(x_cur)
        else:
            x_hat = x_cur

        t_eval = t_hat * \
            add_dimensions(torch.ones(
                x.shape[0], device=x.device), len(x.shape) - 1)
        D = denoiser(x_hat, t_eval, y)
        d_cur = (x_hat - D) / t_hat
        x_next = x_hat + (t1 - t_hat) * d_cur

        if i < num_steps - 1:
            t_eval = t1 * \
                add_dimensions(torch.ones(
                    x.shape[0], device=x.device), len(x.shape) - 1)
            D = denoiser(x_next, t_eval, y)
            d_prime = (x_next - D) / t1
            x_next = x_hat + (t1 - t_hat) * (.5 * d_cur + .5 * d_prime)

        x_cur = x_next

    return D


def ddim_sampler(x, y, denoiser, num_steps, tmin, tmax, rho, guid_scale=None, stochastic=False, **kwargs):
    t_steps = torch.linspace(tmax ** (1. / rho), tmin **
                             (1. / rho), steps=num_steps, device=x.device) ** rho
    x = x * t_steps[0]

    if guid_scale is not None:
        denoiser = guidance_wrapper(denoiser, guid_scale)

    for _, (t0, t1) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        dt = t1 - t0
        t_eval = t0 * \
            add_dimensions(torch.ones(
                x.shape[0], device=x.device), len(x.shape) - 1)

        D = denoiser(x, t_eval, y)

        if stochastic:
            x = x + 2 * dt * (x - D) / t0
            x = x + torch.sqrt(2 * dt.abs() * t0) * \
                torch.randn_like(x, device=x.device)
        else:
            f = (x - D) / t0
            x = x + dt * f

    t_eval = t_steps[-1] * \
        add_dimensions(torch.ones(
            x.shape[0], device=x.device), len(x.shape) - 1)
    x = denoiser(x, t_eval, y)
    return x
