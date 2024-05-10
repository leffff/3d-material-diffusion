import torch
from torch.nn import functional as F


def diffusion_loss(model, noise, x_1, t, noise_scheduler):
    x_t = noise_scheduler.add_noise(x_1, noise, t)

    noise_pred = model(x_t, t)
    
    return F.mse_loss(noise_pred, noise, reduction="mean")


def flow_matching_loss(model, x_0, x_1, t):
    t = t.reshape((t.shape[0], 1, 1, 1, 1))
    
    x_t = t * x_1 + (1 - t) * x_0
    t = t.flatten()
    path_pred = model(x_t, t)
    
    return F.mse_loss(path_pred, (x_1 - x_0), reduction="mean")
