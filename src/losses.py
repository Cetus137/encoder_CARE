import torch
import torch.nn.functional as F
from torch import nn


def kl_divergence(mu, logvar):
    # returns KL divergence per batch (sum over latent dims)
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()


def l1_loss(x, y):
    return F.l1_loss(x, y)


def dice_loss(pred, target, eps=1e-6):
    # pred and target are probabilities (0-1)
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    inter = (pred * target).sum(1)
    denom = pred.sum(1) + target.sum(1)
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice.mean()


def batch_mean_l1(z1, z2):
    # align means across batch for domain invariance
    m1 = z1.mean(dim=0)
    m2 = z2.mean(dim=0)
    return F.l1_loss(m1, m2)


def mmd_rbf(x, y, sigma_list=[1, 2, 4, 8]):
    # simple RBF MMD between two batches
    xx = _rbf_kernel(x, x, sigma_list)
    yy = _rbf_kernel(y, y, sigma_list)
    xy = _rbf_kernel(x, y, sigma_list)
    mmd = xx.mean() + yy.mean() - 2 * xy.mean()
    return mmd


def _rbf_kernel(x, y, sigma_list):
    # x: (B, D), y: (B', D)
    xx = x.unsqueeze(1)  # B x 1 x D
    yy = y.unsqueeze(0)  # 1 x B' x D
    dist = ((xx - yy) ** 2).sum(-1)
    K = 0
    for s in sigma_list:
        K = K + torch.exp(-dist / (2 * s ** 2))
    return K


def segmentation_entropy_loss(seg, eps=1e-6):
    """Encourage segmentation to have diversity (not all 0s or all 1s).
    Returns negative entropy - we want to maximize entropy to prevent collapse.
    """
    # seg: (B, 1, H, W) - probabilities after sigmoid
    p = seg.mean()  # average probability across all pixels
    # Binary entropy: -p*log(p) - (1-p)*log(1-p)
    # We want this to be high (close to 0.5 is maximum entropy)
    entropy = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
    # Return negative entropy as loss (minimizing this maximizes entropy)
    return -entropy


class TotalLoss:
    def __init__(self, weights=None):
        default = dict(
            rec=100.0,
            kl=0.1,  # Reduced from 1.0 to prevent KL from dominating
            align=10.0,
            cross=10.0,
            seg=5.0,
            perc=1.0,
            seg_entropy=1.0,  # Entropy regularization to prevent constant segmentations
        )
        if weights is None:
            weights = default
        self.w = weights

    def __call__(self, preds, targets, extra=None, kl_anneal=1.0):
        # preds: dict with xhat_C, xhat_D, x_CD, etc.
        # targets: dict with x_C, x_D, S_D, ...
        loss_rec = l1_loss(preds['xhat_C'], targets['x_C']) + l1_loss(preds['xhat_D'], targets['x_D'])

        kl_s = preds.get('kl_s', 0)
        kl_t = preds.get('kl_t', 0)
        loss_kl = (kl_s + kl_t) * 0.5 * kl_anneal

        # alignment over shape latents
        loss_align = batch_mean_l1(preds['mu_s_C'], preds['mu_s_D'])

        # cross reconstruction: segmentation-level L1
        loss_cross = l1_loss(preds['S_x_CD'], targets['S_x_D'])

        loss_seg = dice_loss(preds['S_x_CD'], targets['S_x_D'])
        
        # Entropy regularization to prevent segmentation collapse
        loss_seg_entropy = (segmentation_entropy_loss(preds['S_x_CD']) + 
                           segmentation_entropy_loss(targets['S_x_D'])) * 0.5

        # optional perceptual: use encoded features
        if 'feat_x_CD' in preds and 'feat_x_D' in preds:
            loss_perc = l1_loss(preds['feat_x_CD'], preds['feat_x_D'])
        else:
            loss_perc = torch.tensor(0.0, device=loss_rec.device)

        total = (
            self.w['rec'] * loss_rec
            + self.w['kl'] * loss_kl
            + self.w['align'] * loss_align
            + self.w['cross'] * loss_cross
            + self.w['seg'] * loss_seg
            + self.w.get('seg_entropy', 0.0) * loss_seg_entropy
            + self.w['perc'] * loss_perc
        )

        return dict(total=total, rec=loss_rec, kl=loss_kl, align=loss_align, cross=loss_cross, seg=loss_seg, 
                   seg_entropy=loss_seg_entropy, perc=loss_perc)
