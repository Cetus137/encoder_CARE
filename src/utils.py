import os
import torch
import torchvision.utils as vutils
import numpy as np


def normalize_tensor(x):
    # x in [0,1], keep as is; could add mean/std later
    return x


def denormalize_tensor(x):
    # x tensor [B,C,H,W] -> numpy [H,W]
    x = x.detach().cpu()
    return x


def save_checkpoint(state, out_dir, name='checkpoint.pth'):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    torch.save(state, path)
    return path


def load_checkpoint(path, device='cpu'):
    return torch.load(path, map_location=device)


def make_grid(x, nrow=4):
    # x tensor
    return vutils.make_grid(x, nrow=nrow, normalize=True, scale_each=True)


def save_epoch_visuals(out_dir, epoch, images_dict, keys=None, n_samples=4, nrow=None):
    """Save a visualization PNG that shows workflows per sample.

    images_dict: dict of name -> tensor (B,C,H,W) or (B,H,W)
    keys: order of keys to display per sample
    n_samples: how many samples (rows) to show
    nrow: number of images per row (defaults to len(keys))
    """
    os.makedirs(out_dir, exist_ok=True)
    if keys is None:
        keys = list(images_dict.keys())
    if nrow is None:
        nrow = len(keys)

    # prepare list: for each sample, append the tensors in keys order
    images_list = []
    # ensure tensors are CPU and have shape (C,H,W)
    for i in range(n_samples):
        for k in keys:
            t = images_dict.get(k)
            if t is None:
                continue
            # t can be (B,1,H,W) or (B,H,W)
            if t.dim() == 4:
                img = t[i].detach().cpu()
            elif t.dim() == 3:
                img = t[i].unsqueeze(0).detach().cpu()
            else:
                raise ValueError(f"Unsupported tensor shape for key {k}: {t.shape}")

            # Ensure float and clamp to [0,1]
            img = img.float()
            img = torch.clamp(img, 0.0, 1.0)

            # Convert single-channel to 3-channel for better visibility
            if img.size(0) == 1:
                img = img.repeat(3, 1, 1)

            images_list.append(img)

    if len(images_list) == 0:
        return None

    grid = vutils.make_grid(images_list, nrow=nrow, normalize=True, scale_each=True)
    out_path = os.path.join(out_dir, f'epoch_{epoch:04d}_viz.png')
    vutils.save_image(grid, out_path)
    return out_path
