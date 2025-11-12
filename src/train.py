import os
import torch
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import json

from .models import Encoder, Decoder, SegmentationUNet, reparameterize
from .losses import TotalLoss, kl_divergence
from .dataset import get_dataloaders
from .utils import save_checkpoint
from .utils import save_epoch_visuals


def train_model(
    data_root="./data",
    clean_dir='clean',
    degraded_dir='degraded',
    epochs=100,
    batch_size=8,
    lr=2e-4,
    kl_anneal_epochs=10,
    device=None,
    save_interval=10,
    out_dir="./outputs",
    image_size=128,
    normalization='global',
    pmin=0.5,
    pmax=99.5,
    max_samples=1000000,
    norm_cache=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # models
    enc = Encoder(in_ch=1).to(device)
    dec = Decoder(out_ch=1, output_size=image_size).to(device)
    seg = SegmentationUNet(in_ch=1).to(device)

    params = list(enc.parameters()) + list(dec.parameters()) + list(seg.parameters())
    opt = optim.Adam(params, lr=lr)

    dataloader = get_dataloaders(root=data_root, batch_size=batch_size, image_size=image_size,
                                normalization=normalization, pmin=pmin, pmax=pmax, max_samples=max_samples, norm_cache=norm_cache,
                                clean_dir=clean_dir, degraded_dir=degraded_dir)

    loss_fn = TotalLoss()

    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        enc.train(); dec.train(); seg.train()
        running = {}
        for i, (x_C, x_D) in enumerate(dataloader):
            x_C = x_C.to(device)
            x_D = x_D.to(device)

            # Ensure single-channel inputs: if tensors are (B,H,W) add channel dim.
            # If multi-channel tensors are provided, raise an informative error.
            if x_C.dim() == 3:
                x_C = x_C.unsqueeze(1)
            if x_D.dim() == 3:
                x_D = x_D.unsqueeze(1)
            if x_C.size(1) > 1:
                raise ValueError(f"x_C has {x_C.size(1)} channels; encoder_CARE expects single-channel images. Please provide single-channel TIFFs.")
            if x_D.size(1) > 1:
                raise ValueError(f"x_D has {x_D.size(1)} channels; encoder_CARE expects single-channel images. Please provide single-channel TIFFs.")

            # Encode
            (mu_s_C, logv_s_C), (mu_t_C, logv_t_C), feat_C = enc(x_C)
            (mu_s_D, logv_s_D), (mu_t_D, logv_t_D), feat_D = enc(x_D)

            z_s_C = reparameterize(mu_s_C, logv_s_C)
            z_t_C = reparameterize(mu_t_C, logv_t_C)
            z_s_D = reparameterize(mu_s_D, logv_s_D)
            z_t_D = reparameterize(mu_t_D, logv_t_D)

            # Reconstructions
            xhat_C = dec(z_s_C, z_t_C, domain=1)
            xhat_D = dec(z_s_D, z_t_D, domain=0)

            # cross-domain style means
            zt_C_mean = z_t_C.mean(dim=0, keepdim=True).expand_as(z_t_C)
            zt_D_mean = z_t_D.mean(dim=0, keepdim=True).expand_as(z_t_D)

            x_CD = dec(z_s_D, zt_C_mean, domain=1)
            x_DC = dec(z_s_C, zt_D_mean, domain=0)

            # segmentation preds
            S_D = seg(x_D)
            S_x_CD = seg(x_CD)

            # compute losses
            preds = dict(
                xhat_C=xhat_C,
                xhat_D=xhat_D,
                x_CD=x_CD,
                x_DC=x_DC,
                S_x_CD=S_x_CD,
                mu_s_C=mu_s_C,
                mu_s_D=mu_s_D,
                feat_x_CD=feat_D,  # placeholder: optional
                feat_x_D=feat_D,
                kl_s=kl_divergence(mu_s_C, logv_s_C) + kl_divergence(mu_s_D, logv_s_D),
                kl_t=kl_divergence(mu_t_C, logv_t_C) + kl_divergence(mu_t_D, logv_t_D),
            )

            targets = dict(x_C=x_C, x_D=x_D, S_x_D=S_D)

            kl_anneal = min(1.0, epoch / max(1.0, kl_anneal_epochs))
            loss_dict = loss_fn(preds, targets, kl_anneal=kl_anneal)
            total_loss = loss_dict['total']

            opt.zero_grad()
            total_loss.backward()
            opt.step()

            if i % 20 == 0:
                print(f"Epoch {epoch}/{epochs} Iter {i} Loss {total_loss.item():.4f}")

        # at epoch end, save visuals from the last batch
        try:
            # pick up to 4 samples from the last processed batch
            n_vis = min(4, x_D.size(0))
            images_dict = dict(
                original=x_C[:n_vis].detach().cpu(),
                degraded=x_D[:n_vis].detach().cpu(),
                recon=xhat_D[:n_vis].detach().cpu(),
                restored=x_CD[:n_vis].detach().cpu(),
                seg_degraded=S_D[:n_vis].detach().cpu(),
                seg_restored=S_x_CD[:n_vis].detach().cpu(),
            )
            keys = ['original', 'degraded', 'recon', 'restored', 'seg_degraded', 'seg_restored']
            vis_path = save_epoch_visuals(out_dir, epoch, images_dict, keys=keys, n_samples=n_vis, nrow=len(keys))
            if vis_path:
                print(f"Saved visualization: {vis_path}")
            # compute and display image statistics for original and degraded (post-normalization)
            try:
                def _stats(tensor):
                    arr = tensor.detach().cpu().numpy()
                    if arr.ndim == 3:
                        arr = arr.squeeze(1)
                    # flatten across batch and spatial dims
                    flat = arr.reshape(-1)
                    flat = flat[np.isfinite(flat)]
                    if flat.size == 0:
                        return None
                    stats = {}
                    stats['min'] = float(np.min(flat))
                    stats['max'] = float(np.max(flat))
                    stats['mean'] = float(np.mean(flat))
                    stats['std'] = float(np.std(flat))
                    for p in [0.5, 1.0, 5.0, 50.0, 95.0, 99.0, 99.5]:
                        stats[f'p{p}'] = float(np.percentile(flat, p))
                    return stats

                stats_orig = _stats(x_C[:n_vis])
                stats_deg = _stats(x_D[:n_vis])
                print(f"Epoch {epoch} stats (original): {stats_orig}")
                print(f"Epoch {epoch} stats (degraded):  {stats_deg}")

                # save stats to json
                stats_path = os.path.join(out_dir, f'epoch_{epoch:04d}_stats.json')
                with open(stats_path, 'w') as f:
                    json.dump({'epoch': epoch, 'original': stats_orig, 'degraded': stats_deg}, f, indent=2)
                print(f"Saved stats: {stats_path}")
            except Exception as e:
                print(f"Failed to compute/save stats: {e}")
            # NOTE: we intentionally do NOT save degraded TIFFs anymore (user requested no TIFF exports).
        except Exception as e:
            print(f"Visualization failed: {e}")

        # save checkpoint
        if epoch % save_interval == 0 or epoch == epochs:
            state = dict(epoch=epoch, enc=enc.state_dict(), dec=dec.state_dict(), seg=seg.state_dict(), opt=opt.state_dict())
            save_checkpoint(state, out_dir, name=f'checkpoint_epoch_{epoch}.pth')

    return dict(enc=enc, dec=dec, seg=seg)
