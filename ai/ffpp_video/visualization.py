"""Draw temporal contribution bars and Grad-CAM overlays for the model input."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .constants import CONTACT_SHEET_FRAMES, IMAGENET_MEAN, IMAGENET_STD, NUM_FRAMES
from .exceptions import XaiError


def representative_indices(fallback_flags: list[bool], count: int = CONTACT_SHEET_FRAMES) -> list[int]:
    """Pick evenly spaced frames, then keep every fallback frame in time order."""
    if len(fallback_flags) != NUM_FRAMES:
        raise XaiError(f"Expected {NUM_FRAMES} fallback flags, got {len(fallback_flags)}.")
    if count < 1 or count > NUM_FRAMES:
        raise XaiError(f"Contact sheet count must be between 1 and {NUM_FRAMES}.")
    chosen = []
    for slot in range(count):
        index = min(NUM_FRAMES - 1, int(slot * NUM_FRAMES / count))
        if index not in chosen:
            chosen.append(index)
    for index, is_fallback in enumerate(fallback_flags):
        if is_fallback and index not in chosen:
            replaceable = [item for item in chosen if not fallback_flags[item]]
            if not replaceable:
                break
            replaced = min(replaceable, key=lambda item: (abs(item - index), item))
            chosen.remove(replaced)
            chosen.append(index)
    return sorted(chosen)


def model_input_rgb(frame_chw: np.ndarray) -> np.ndarray:
    """Undo ImageNet normalization so the overlay matches the tensor the model saw."""
    array = np.transpose(frame_chw, (1, 2, 0)).astype(np.float32)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    restored = (array * std + mean) * 255.0
    return np.clip(restored, 0, 255).astype(np.uint8)


def overlay_heatmap(rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Blend a [0, 1] heatmap onto the RGB model input."""
    if rgb.shape[:2] != heatmap.shape:
        raise XaiError(f"Heatmap {heatmap.shape} does not match input {rgb.shape[:2]}.")
    heat = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 0.55, color, 0.45, 0)


def save_temporal_plot(path: Path, frames: list[dict]) -> None:
    """Bar chart of normalized temporal contribution in frame order."""
    if len(frames) != NUM_FRAMES:
        raise XaiError("Temporal plot expected 16 frames.")
    width, height = 960, 420
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(
        image,
        "frames with higher relative contribution",
        (24, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    left, right, top, bottom = 50, 930, 60, 360
    plot_width = right - left
    plot_height = bottom - top
    values = [float(frame["temporal_importance"]) for frame in frames]
    peak = max(values + [1e-8])
    slot = plot_width / NUM_FRAMES
    for index, frame in enumerate(frames):
        bar_height = int(plot_height * (values[index] / peak))
        x0 = int(left + index * slot + 6)
        x1 = int(left + (index + 1) * slot - 6)
        color = (40, 90, 170) if frame["used_full_frame_fallback"] else (40, 110, 70)
        cv2.rectangle(image, (x0, bottom - bar_height), (x1, bottom), color, thickness=-1)
        cv2.putText(
            image,
            str(frame["frame_index"]),
            (x0 - 2, bottom + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        image,
        "x = frame index    y = normalized temporal contribution    blue = full-frame fallback",
        (24, height - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise XaiError(f"Could not write {path}.")


def save_frame_gradcam(path: Path, frame_chw: np.ndarray, heatmap: np.ndarray) -> None:
    """Write one Grad-CAM overlay for the model input of a single sampled frame."""

    overlay = overlay_heatmap(model_input_rgb(frame_chw), heatmap)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
        raise XaiError(f"Could not write {path.name}.")


def save_contact_sheet(path: Path, model_inputs: np.ndarray, heatmaps: torch.Tensor, frames: list[dict]) -> None:
    """Show model inputs and overlays for representative frames, in time order."""
    flags = [bool(frame["used_full_frame_fallback"]) for frame in frames]
    selected = representative_indices(flags)
    tile_w, tile_h = 224, 224
    label_h = 58
    pad = 8
    columns = 4
    rows = int(np.ceil(len(selected) / columns))
    sheet_w = columns * (tile_w * 2 + pad) + pad
    sheet_h = rows * (tile_h + label_h + pad) + pad
    sheet = np.full((sheet_h, sheet_w, 3), 245, dtype=np.uint8)
    heat = heatmaps.detach().cpu().numpy()
    for position, index in enumerate(selected):
        row = position // columns
        column = position % columns
        x = pad + column * (tile_w * 2 + pad)
        y = pad + row * (tile_h + label_h + pad)
        rgb = model_input_rgb(model_inputs[index])
        overlay = overlay_heatmap(rgb, heat[index])
        sheet[y + label_h : y + label_h + tile_h, x : x + tile_w] = rgb
        sheet[y + label_h : y + label_h + tile_h, x + tile_w : x + tile_w * 2] = overlay
        frame = frames[index]
        status = "full-frame fallback" if frame["used_full_frame_fallback"] else "face crop"
        cv2.putText(
            sheet,
            f"frame {frame['frame_index']}  t={frame['timestamp']:.2f}s  {status}",
            (x, y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            sheet,
            f"relative contribution {frame['temporal_importance']:.3f}",
            (x, y + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)):
        raise XaiError(f"Could not write {path}.")
