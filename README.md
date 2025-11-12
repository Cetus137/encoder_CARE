# Non-Adversarial Shared-Encoder Disentangled VAE for Unpaired Fluorescence Restoration

This small PyTorch project implements a shared-encoder VAE that disentangles structural (shape) and style latents for restoration of degraded fluorescence microscopy images without any GANs.

Files
- `models.py` - Encoder, Decoder with FiLM conditioning, Segmentation U-Net
- `losses.py` - KL, alignment, Dice, total loss aggregation
- `dataset.py` - Unpaired loader reading `data/clean/` and `data/degraded/`
- `train.py` - `train_model(...)` function entrypoint (import and call from notebook)
- `utils.py` - helpers for checkpointing and visualization

Quick start (in a notebook):

```python
from encoder_CARE.train import train_model

train_model(data_root='./data', epochs=50, batch_size=8, device='cuda')
```

Data layout
- `data/clean/` : clean fluorescence images (png, jpg, tiff)
- `data/degraded/` : degraded images

Notes
- This implementation is intentionally minimal and designed to be imported from Python code or notebooks. There is no CLI or argparse.
- Adjust image size, channels, and hyperparameters in the code as needed for your dataset.
