"""Simple runner to call train_model from a script or terminal.

No argparse — edit the `cfg` dictionary below to change parameters, then run this file.

Run directly:

    python main.py

This script imports from the `src` package (made a package by `src/__init__.py`),
so it works when run as a script or when invoked with `python -m`.
"""

try:
    # Prefer direct import from the `src` package so running `python main.py`
    # works without relying on module-mode execution (`python -m ...`).
    from src.train import train_model
except ModuleNotFoundError as e:
    # Friendly message when core runtime deps (like torch) are missing.
    print("Failed to import training code — missing dependency or package import error:", e)
    print("Make sure you run this from the project root and that your environment has the required packages installed (see requirements.txt).")
    raise
except Exception:
    # Fallback to package export if present (keeps compatibility with
    # `python -m encoder_CARE.main` invocation).
    from encoder_CARE import train_model


def main():
    # Edit these values directly. This is a simple dict-based config (no env vars).
    # Use relative paths so running the script from the project root is convenient.
    cfg = {
        'data_root': '/Users/ewheeler/encoder_CARE/data',
        'clean_dir': 'clean',
        'degraded_dir': 'degraded',
        'epochs': 100,
        'batch_size': 32,
        'lr': 2e-4,
        'kl_anneal_epochs': 10,
        'device': None,  # set to 'cuda' or 'cpu' to override automatic selection
        'save_interval': 10,
        'out_dir': './outputs',
        'image_size': 128,
    }

    print("Starting training with config:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    train_model(**cfg)


if __name__ == '__main__':
    main()
