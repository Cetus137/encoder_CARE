import os
import random
import torch
import tifffile as tiff
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class UnpairedImageDataset(Dataset):
    def __init__(self, root='./data', clean_dir='clean', degraded_dir='degraded',
                 image_size=64, augment=True,
                 normalization='global', pmin=0.5, pmax=99.5, max_samples=1000000, norm_cache=None):
        self.clean_paths = self._list_images(os.path.join(root, clean_dir))
        self.deg_paths = self._list_images(os.path.join(root, degraded_dir))
        self.size = image_size
        self.augment = augment
        self.normalization = normalization
        self.pmin = float(pmin)
        self.pmax = float(pmax)
        self.max_samples = int(max_samples)
        self.norm_cache = norm_cache or os.path.join(root, 'norm_stats.npz')

        print(f"UnpairedImageDataset initialized:")
        print(f"  Clean images: {len(self.clean_paths)} from {os.path.join(root, clean_dir)}")
        print(f"  Degraded images: {len(self.deg_paths)} from {os.path.join(root, degraded_dir)}")
        print(f"  Normalization mode: {self.normalization}")

        # compute or load global normalization if requested
        self.global_min = None
        self.global_max = None
        if self.normalization == 'global':
            self._prepare_global_normalization()


    def _list_images(self, p):
        if not os.path.exists(p):
            return []
        exts = ['.tif', '.tiff']
        files = [os.path.join(p, f) for f in sorted(os.listdir(p)) if os.path.splitext(f)[1].lower() in exts]
        return files

    def _prepare_global_normalization(self):
        # Try to load cache
        if os.path.exists(self.norm_cache):
            try:
                d = np.load(self.norm_cache)
                self.global_min = float(d['pmin'])
                self.global_max = float(d['pmax'])
                return
            except Exception:
                pass

        # gather sample pixels up to max_samples
        paths = list(self.clean_paths) + list(self.deg_paths)
        sample_list = []
        collected = 0
        per_image_limit = max(1, int(self.max_samples / max(1, len(paths))))
        for p in paths:
            try:
                arr = tiff.imread(p)
            except Exception:
                continue
            arr = np.asarray(arr)
            # collapse to grayscale if needed
            if arr.ndim == 3:
                if arr.shape[2] in (3, 4):
                    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
                    arr = (0.2989 * r + 0.5870 * g + 0.1140 * b)
                else:
                    arr = arr[..., 0]

            flat = arr.ravel()
            if flat.size == 0:
                continue
            # sample up to per_image_limit random pixels
            if flat.size > per_image_limit:
                idx = np.random.choice(flat.size, size=per_image_limit, replace=False)
                samp = flat[idx]
            else:
                samp = flat

            sample_list.append(samp)
            collected += samp.size
            if collected >= self.max_samples:
                break

        if len(sample_list) == 0:
            # fallback
            self.global_min = 0.0
            self.global_max = 1.0
        else:
            all_samps = np.concatenate(sample_list)
            pmin_val, pmax_val = np.percentile(all_samps, [self.pmin, self.pmax])
            # avoid degenerate ranges
            if pmax_val <= pmin_val:
                pmin_val, pmax_val = float(all_samps.min()), float(all_samps.max())
                if pmax_val <= pmin_val:
                    pmin_val, pmax_val = 0.0, 1.0

            self.global_min = float(pmin_val)
            self.global_max = float(pmax_val)

        try:
            np.savez(self.norm_cache, pmin=self.global_min, pmax=self.global_max)
        except Exception:
            pass

    def __len__(self):
        return max(len(self.clean_paths), len(self.deg_paths))

    def __getitem__(self, idx):
        c = self.clean_paths[idx % len(self.clean_paths)] if self.clean_paths else None
        d = self.deg_paths[idx % len(self.deg_paths)] if self.deg_paths else None
        
        # Debug: print first item
        if idx == 0:
            print(f"DEBUG __getitem__(0):")
            print(f"  clean path: {c}")
            print(f"  degraded path: {d}")
        
        def _load_tiff_to_tensor(path):
            # Read TIFF into numpy and convert to float32 tensor with range [0,1]
            arr = tiff.imread(path)
            arr = np.asarray(arr)

            # Collapse channels to grayscale if needed
            if arr.ndim == 3:
                if arr.shape[2] in (3, 4):
                    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
                    arr = (0.2989 * r + 0.5870 * g + 0.1140 * b)
                else:
                    arr = arr[..., 0]

            arr = arr.astype(np.float32)
            if arr.size == 0:
                arr = np.zeros((self.size, self.size), dtype=np.float32)

            # Normalization options:
            # - 'none': assume image is already in 0..1 (float32/float64). We will
            #           clip to [0,1] and leave values unchanged otherwise.
            # - 'per_image': rescale each image to [0,1] based on its min/max.
            # - 'global': clip to the precomputed global percentiles and scale.
            if self.normalization == 'none':
                # If the data is float and within 0..1 this preserves it; for
                # integer types we still convert to float and scale by type
                # range if needed — but here we assume user provided 0..1 floats.
                # To be robust, if the image appears to be in a larger range
                # (e.g., 0-255 ints mistakenly saved as floats) we fall back to
                # per-image normalization.
                vmin = float(arr.min())
                vmax = float(arr.max())
                if vmax <= vmin:
                    # degenerate: make zeros
                    arr = arr * 0.0
                elif vmax <= 1.0 and vmin >= 0.0:
                    # already 0..1 float — just clip small numerical errors
                    arr = np.clip(arr, 0.0, 1.0)
                else:
                    # Unexpected range — fallback to per-image normalization
                    a_min = vmin
                    a_max = vmax
                    if a_max > a_min:
                        arr = (arr - a_min) / (a_max - a_min)
                    else:
                        arr = arr * 0.0

            elif self.normalization == 'per_image':
                # Use per-image percentile clipping followed by scaling. This
                # is more robust to outliers than min/max scaling. The
                # percentiles used are `self.pmin` and `self.pmax` (defaults
                # 0.5/99.5). If percentiles collapse to a degenerate range we
                # fall back to strict min/max scaling and finally to zeros.
                try:
                    lo = float(np.percentile(arr, self.pmin))
                    hi = float(np.percentile(arr, self.pmax))
                except Exception:
                    lo = float(arr.min())
                    hi = float(arr.max())

                if hi > lo:
                    arr = np.clip(arr, lo, hi)
                    arr = (arr - lo) / (hi - lo)
                else:
                    # fallback to min/max
                    a_min = float(arr.min())
                    a_max = float(arr.max())
                    if a_max > a_min:
                        arr = (arr - a_min) / (a_max - a_min)
                    else:
                        arr = arr * 0.0
            else:
                # global normalization by clipping to percentiles and scaling
                if self.global_min is None or self.global_max is None:
                    # fallback to per-image
                    a_min = float(arr.min())
                    a_max = float(arr.max())
                    if a_max > a_min:
                        arr = (arr - a_min) / (a_max - a_min)
                    else:
                        arr = arr * 0.0
                else:
                    lo = self.global_min
                    hi = self.global_max
                    if hi <= lo:
                        arr = arr * 0.0
                    else:
                        arr = np.clip(arr, lo, hi)
                        arr = (arr - lo) / (hi - lo)

            t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()  # 1,1,H,W
            # resize to target size
            t = F.interpolate(t, size=(self.size, self.size), mode='bilinear', align_corners=False)
            t = t.squeeze(0)  # 1,H,W
            return t

        if c is not None:
            t_c = _load_tiff_to_tensor(c)
        else:
            t_c = torch.zeros((1, self.size, self.size), dtype=torch.float32)

        if d is not None:
            t_d = _load_tiff_to_tensor(d)
        else:
            t_d = torch.zeros((1, self.size, self.size), dtype=torch.float32)

        # augment with random flips (tensor-level)
        if self.augment:
            if random.random() < 0.5:
                t_c = torch.flip(t_c, dims=[2])
                t_d = torch.flip(t_d, dims=[2])
            if random.random() < 0.5:
                t_c = torch.flip(t_c, dims=[1])
                t_d = torch.flip(t_d, dims=[1])

        return t_c, t_d

def get_dataloaders(root='./data', clean_dir='clean', degraded_dir='degraded', batch_size=8, image_size=64, augment=True, num_workers=8,
                    normalization='global', pmin=0.5, pmax=99.5, max_samples=1000000, norm_cache=None, pin_memory=True):
    ds = UnpairedImageDataset(root=root, clean_dir=clean_dir, degraded_dir=degraded_dir, image_size=image_size, augment=augment,
                              normalization=normalization, pmin=pmin, pmax=pmax, max_samples=max_samples, norm_cache=norm_cache)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=pin_memory)
    return loader
