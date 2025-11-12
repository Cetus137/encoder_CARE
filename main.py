"""Simple runner to call train_model from a script or terminal.

No argparse — edit the `cfg` dictionary below to change parameters, then run this file.
"""

from encoder_CARE.train import train_model

def main():
    # Edit these values directly. This is a simple dict-based config (no env vars).
    cfg = {
        'data_root': '/Users/ewheeler/encoder_CARE/data',
        'epochs': 100,
        'batch_size': 32,
        'lr': 2e-4,
        'kl_anneal_epochs': 10,
        'device': None,  # set to 'cuda' or 'cpu' to override automatic selection
        'save_interval': 10,
        'out_dir': '/Users/ewheeler/encoder_CARE/outputs',
        'image_size': 128

    }

    print("Starting training with config:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    train_model(**cfg)


if __name__ == '__main__':
    main()
