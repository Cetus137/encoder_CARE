import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    """Shared encoder producing shape and style latent params.
    Produces (mu_shape, logvar_shape), (mu_style, logvar_style)
    Also returns an intermediate feature map for optional perceptual loss.
    """
    def __init__(self, in_ch=1, base_ch=32, z_shape_dim=256, z_style_dim=128):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.pool = nn.MaxPool2d(2)

        # We'll adaptively pool features to a fixed spatial size (8x8) so the FC
        # layer input size is constant regardless of input image resolution.
        feat_dim = base_ch * 4 * 8 * 8
        self.fc = nn.Linear(feat_dim, base_ch * 8)

        self.fc_mu_shape = nn.Linear(base_ch * 8, z_shape_dim)
        self.fc_logvar_shape = nn.Linear(base_ch * 8, z_shape_dim)
        self.fc_mu_style = nn.Linear(base_ch * 8, z_style_dim)
        self.fc_logvar_style = nn.Linear(base_ch * 8, z_style_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        h = self.enc1(x)
        h = self.pool(h)
        h = self.enc2(h)
        h = self.pool(h)
        h = self.enc3(h)
        h = self.pool(h)
        feat = h

        # adaptive pool to 8x8 so fc input shape is fixed
        pooled = F.adaptive_avg_pool2d(feat, (8, 8))
        b, c, h_, w_ = pooled.shape
        flat = pooled.view(b, -1)
        hfc = F.relu(self.fc(flat))

        mu_s = self.fc_mu_shape(hfc)
        logvar_s = self.fc_logvar_shape(hfc)
        mu_t = self.fc_mu_style(hfc)
        logvar_t = self.fc_logvar_style(hfc)

        return (mu_s, logvar_s), (mu_t, logvar_t), feat


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class FiLM(nn.Module):
    def __init__(self, channels, style_dim):
        super().__init__()
        self.gamma = nn.Linear(style_dim, channels)
        self.beta = nn.Linear(style_dim, channels)

    def forward(self, x, style):
        # x: (B, C, H, W), style: (B, style_dim)
        # compute raw projections
        gamma_lin = self.gamma(style)
        beta_lin = self.beta(style)

        gamma = gamma_lin.unsqueeze(-1).unsqueeze(-1)
        beta = beta_lin.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta


class Decoder(nn.Module):
    """Conditional decoder using FiLM conditioning on style and domain embedding"""
    def __init__(self, out_ch=1, base_ch=32, z_shape_dim=256, z_style_dim=128, domain_embed_dim=8, output_size=None):
        super().__init__()
        self.output_size = output_size
        self.fc = nn.Linear(z_shape_dim + z_style_dim + domain_embed_dim, base_ch * 8)

        self.up1 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 4, stride=2, padding=1)
        self.conv1 = ConvBlock(base_ch * 4, base_ch * 4)
        self.film1 = FiLM(base_ch * 4, z_style_dim + domain_embed_dim)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1)
        self.conv2 = ConvBlock(base_ch * 2, base_ch * 2)
        self.film2 = FiLM(base_ch * 2, z_style_dim + domain_embed_dim)

        self.up3 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1)
        self.conv3 = ConvBlock(base_ch, base_ch)
        self.film3 = FiLM(base_ch, z_style_dim + domain_embed_dim)

        self.final = nn.Conv2d(base_ch, out_ch, 1)

        # domain embedding: two learnable vectors
        self.domain_embeddings = nn.Embedding(2, domain_embed_dim)
        # Optional extra upsampling to reach a desired native output size
        self.extra_ups = nn.ModuleList()
        self.extra_films = nn.ModuleList()
        self.extra_convs = nn.ModuleList()
        if self.output_size is not None:
            try:
                target = int(self.output_size)
                base_out = 8  # current decoder produces 8x8 before interpolation
                if target > base_out:
                    ratio = target // base_out
                    # compute number of extra doubling steps (floor)
                    n_extra = int(np.floor(np.log2(ratio))) if ratio >= 1 else 0
                    # create n_extra upsampling blocks that keep channel size at base_ch
                    for _ in range(n_extra):
                        self.extra_ups.append(nn.ConvTranspose2d(base_ch, base_ch, 4, stride=2, padding=1))
                        self.extra_convs.append(ConvBlock(base_ch, base_ch))
                        self.extra_films.append(FiLM(base_ch, z_style_dim + domain_embed_dim))
            except Exception:
                # ignore failures and fall back to interpolation behavior
                pass

    def forward(self, z_shape, z_style, domain):
        # domain: 0 or 1 or tensor
        if isinstance(domain, int):
            domain = torch.tensor([domain], device=z_shape.device).expand(z_shape.size(0))
        domain_emb = self.domain_embeddings(domain.long())

        style_plus_dom = torch.cat([z_style, domain_emb], dim=1)
        h = torch.cat([z_shape, z_style, domain_emb], dim=1)
        h = F.relu(self.fc(h))
        bsize = h.size(0)
        h = h.view(bsize, -1, 1, 1)
        h = self.up1(h)
        h = self.conv1(h)
        h = self.film1(h, style_plus_dom)

        h = self.up2(h)
        h = self.conv2(h)
        h = self.film2(h, style_plus_dom)

        h = self.up3(h)
        h = self.conv3(h)
        h = self.film3(h, style_plus_dom)

        # If extra upsampling blocks exist, apply them to the feature map (before final conv)
        if len(self.extra_ups) > 0:
            for up, conv, film in zip(self.extra_ups, self.extra_convs, self.extra_films):
                h = up(h)
                h = conv(h)
                h = film(h, style_plus_dom)

        out = torch.sigmoid(self.final(h))

        # Finally, if still not the desired size, interpolate as a fallback
        if self.output_size is not None:
            current_h = out.shape[-2]
            if current_h != self.output_size:
                out = F.interpolate(out, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)

        return out
        return out


class SegmentationUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=16):
        super().__init__()
        # small U-Net
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.pool = nn.MaxPool2d(2)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)

        self.outconv = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d2 = self.up2(e3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = torch.sigmoid(self.outconv(d1))
        return out
