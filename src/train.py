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
    kl_weight=0.1,  # KL divergence weight (lower = less regularization)
    rec_weight=100.0,  # Reconstruction loss weight
    align_weight=10.0,  # Alignment loss weight (shape latent consistency)
    cross_weight=10.0,  # Cross-reconstruction segmentation loss weight
    seg_weight=5.0,  # Segmentation dice loss weight
    perc_weight=1.0,  # Perceptual loss weight
    seg_entropy_weight=1.0,  # Segmentation entropy regularization weight
    device=None,
    save_interval=10,
    out_dir="./outputs",
    image_size=128,
    normalization='global',
    pmin=0.5,
    pmax=99.5,
    max_samples=1000000,
    norm_cache=None,
    num_workers=8,  # Number of parallel data loading workers
    pin_memory=True,  # Pin memory for faster GPU transfer
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device == "cuda"  # Enable automatic mixed precision on GPU
    
    if use_amp:
        print("Using automatic mixed precision (AMP) for faster training")

    # models
    enc = Encoder(in_ch=1).to(device)
    dec = Decoder(out_ch=1, output_size=image_size).to(device)
    seg = SegmentationUNet(in_ch=1).to(device)

    params = list(enc.parameters()) + list(dec.parameters()) + list(seg.parameters())
    opt = optim.Adam(params, lr=lr)
    
    # GradScaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Add learning rate scheduler to help escape plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)

    dataloader = get_dataloaders(root=data_root, batch_size=batch_size, image_size=image_size,
                                normalization=normalization, pmin=pmin, pmax=pmax, max_samples=max_samples, norm_cache=norm_cache,
                                clean_dir=clean_dir, degraded_dir=degraded_dir, 
                                num_workers=num_workers, pin_memory=pin_memory)

    # Initialize loss function with configurable weights
    loss_weights = dict(
        rec=rec_weight,
        kl=kl_weight,
        align=align_weight,
        cross=cross_weight,
        seg=seg_weight,
        perc=perc_weight,
        seg_entropy=seg_entropy_weight,
    )
    loss_fn = TotalLoss(weights=loss_weights)
    print(f"Loss weights: {loss_weights}")

    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        enc.train(); dec.train(); seg.train()
        running = {}
        for i, (x_C, x_D) in enumerate(dataloader):
            x_C = x_C.to(device)
            x_D = x_D.to(device)

            # Debug: check data loading on first iteration
            if epoch == 1 and i == 0:
                print(f"DEBUG - Input data stats:")
                print(f"  x_C: shape={x_C.shape}, min={x_C.min():.4f}, max={x_C.max():.4f}, mean={x_C.mean():.4f}")
                print(f"  x_D: shape={x_D.shape}, min={x_D.min():.4f}, max={x_D.max():.4f}, mean={x_D.mean():.4f}")

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

            # Use automatic mixed precision if on GPU
            with torch.cuda.amp.autocast(enabled=use_amp):
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
            if use_amp:
                scaler.scale(total_loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                total_loss.backward()
                opt.step()

            if i % 100 == 0:  # Changed from 20 to 100 to reduce print overhead
                # latent diagnostics (detect posterior collapse)
                try:
                    zs_mean = z_s_C.mean().item(); zs_std = z_s_C.std().item()
                    zt_mean = z_t_C.mean().item(); zt_std = z_t_C.std().item()
                    zs_mean_d = z_s_D.mean().item(); zs_std_d = z_s_D.std().item()
                    zt_mean_d = z_t_D.mean().item(); zt_std_d = z_t_D.std().item()
                    print(f"Epoch {epoch}/{epochs} Iter {i} Loss {total_loss.item():.4f}")
                    
                    # Display each loss component: raw value and weighted contribution
                    print(f"  Loss components (raw -> weighted):")
                    rec_raw = loss_dict['rec'].item()
                    kl_raw = loss_dict['kl'].item()
                    align_raw = loss_dict['align'].item()
                    cross_raw = loss_dict['cross'].item()
                    seg_raw = loss_dict['seg'].item()
                    perc_raw = loss_dict['perc'].item()
                    
                    print(f"    rec:   {rec_raw:.4f} -> {rec_raw * loss_fn.w['rec']:.4f}")
                    print(f"    kl:    {kl_raw:.4f} -> {kl_raw * loss_fn.w['kl']:.4f}")
                    print(f"    align: {align_raw:.4f} -> {align_raw * loss_fn.w['align']:.4f}")
                    print(f"    cross: {cross_raw:.4f} -> {cross_raw * loss_fn.w['cross']:.4f}")
                    print(f"    seg:   {seg_raw:.4f} -> {seg_raw * loss_fn.w['seg']:.4f}")
                    print(f"    perc:  {perc_raw:.4f} -> {perc_raw * loss_fn.w['perc']:.4f}")
                    print(f"    current_lr: {opt.param_groups[0]['lr']:.6f}")
                    
                    print(f"z_shape C mean/std: {zs_mean:.6f} / {zs_std:.6f}")
                    print(f"z_style C mean/std: {zt_mean:.6f} / {zt_std:.6f}")
                    print(f"z_shape D mean/std: {zs_mean_d:.6f} / {zs_std_d:.6f}")
                    print(f"z_style D mean/std: {zt_mean_d:.6f} / {zt_std_d:.6f}")
                except Exception as e:
                    print(f"Failed to compute latent stats: {e}")

        # at epoch end, save visuals from the last batch
        try:
            # pick up to 4 samples from the last processed batch
            n_vis = min(4, x_D.size(0))
            images_dict = dict(
                original=x_C[:n_vis].detach().cpu().clone(),
                degraded=x_D[:n_vis].detach().cpu().clone(),
                recon=xhat_D[:n_vis].detach().cpu().clone(),
                restored=x_CD[:n_vis].detach().cpu().clone(),
                seg_degraded=S_D[:n_vis].detach().cpu().clone(),
                seg_restored=S_x_CD[:n_vis].detach().cpu().clone(),
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
            except Exception as e:
                print(f"Failed to compute/save stats: {e}")
            # NOTE: we intentionally do NOT save degraded TIFFs anymore (user requested no TIFF exports).
        except Exception as e:
            print(f"Visualization failed: {e}")

        # save checkpoint
        if epoch % save_interval == 0 or epoch == epochs:
            state = dict(epoch=epoch, enc=enc.state_dict(), dec=dec.state_dict(), seg=seg.state_dict(), opt=opt.state_dict())
            save_checkpoint(state, out_dir, name=f'checkpoint_epoch_{epoch}.pth')

        # Update learning rate based on reconstruction loss (helps escape plateaus)
        scheduler.step(loss_dict['rec'].item())

    return dict(enc=enc, dec=dec, seg=seg)
