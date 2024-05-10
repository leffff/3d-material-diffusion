import os
import io
import random

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


def show_images(x: torch.Tensor):
    """Given a batch of images x, make a grid and convert to PIL"""
    # print(x.shape)

    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    # x = x * 0.5 + 0.5 # Map from (-1, 1) back to (0, 1)

    grid = torchvision.utils.make_grid(x)
    grid_im = (grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255)

    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im


def show_microstructure(x):
    x = x[:,0,:,:,:].squeeze(1).clip(-1, 1)
    x = (((x + 1) / 2) * 255).long()
    # x = ((x * 0.5 + 0.5) * 255).long()

    sample = x[0].detach().cpu().numpy()
    fig = plt.figure()
    for i in range(x.shape[0]):
        tess = x[i].detach().cpu().numpy()[:, :, :, np.newaxis].repeat(3, axis=3) / 255
        color = np.concatenate([tess, np.ones((32, 32, 32, 1))], axis=3)
        ax = fig.add_subplot(projection='3d')
        ax.voxels(sample, facecolors=color)

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    
    return fig


def seed_everything(seed: int,
                    use_deterministic_algos: bool = False) -> None:
    
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(use_deterministic_algos)
    random.seed(seed)
    