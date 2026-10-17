"""Standalone inference helpers for a prepared RGB localization image."""
from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from ai.localization.config import LocalizationConfig
from ai.localization.model import LocalizationUNet


def resolve_device(preference: str = "auto") -> torch.device:
    key = preference.lower()
    if key == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(key)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def load_localization_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[LocalizationUNet, dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Localization checkpoint not found: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload.get("config", {})
    model = LocalizationUNet(base_channels=int(cfg.get("base_channels", 16)))
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, payload


@torch.inference_mode()
def predict_image(model: torch.nn.Module, image_path: str | Path, *, device: torch.device | str = "cpu", threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Return float probability and uint8 0/1 masks for a prepared 256 image."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    try:
        with Image.open(image_path) as im:
            rgb = im.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Could not read image: {image_path}") from exc
    if rgb.size != (256, 256):
        raise ValueError(f"Expected a prepared 256x256 image, got {rgb.size}")
    arr = np.asarray(rgb, dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    probabilities = torch.sigmoid(model(tensor))[0, 0].cpu().numpy().astype(np.float32)
    binary = (probabilities >= threshold).astype(np.uint8)
    return probabilities, binary


def save_binary_mask(binary_mask: np.ndarray, output_path: str | Path) -> Path:
    """Save a 0/1 mask as a visual grayscale PNG with values 0/255."""
    arr = np.asarray(binary_mask)
    if arr.shape != (256, 256) or not np.isin(arr, (0, 1)).all():
        raise ValueError("binary_mask must be a 256x256 array containing only 0 and 1")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((arr.astype(np.uint8) * 255), mode="L").save(path, format="PNG")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run standalone localization inference on one prepared 256x256 image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=project_root_checkpoint())
    parser.add_argument("--output", type=Path, default=Path("predicted_mask.png"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    device = resolve_device(args.device)
    model, _ = load_localization_checkpoint(args.checkpoint, device)
    probabilities, binary = predict_image(model, args.image, device=device, threshold=args.threshold)
    save_binary_mask(binary, args.output)
    print(f"Saved binary mask: {args.output} (mean probability={probabilities.mean():.6f})")
    return 0


def project_root_checkpoint() -> Path:
    return Path(__file__).resolve().parents[2] / "artifacts" / "checkpoints" / "localization" / "localization_best.pt"


if __name__ == "__main__":
    raise SystemExit(main())
