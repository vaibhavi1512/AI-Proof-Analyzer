"""Fast focused tests for the isolated pixel-level localization module."""
from __future__ import annotations

import csv

import numpy as np
import pytest
import torch
from PIL import Image

from ai.localization.dataset import LocalizationDataset
from ai.localization.inference import load_localization_checkpoint, predict_image, save_binary_mask
from ai.localization.losses import BCEDiceLoss
from ai.localization.metrics import segmentation_metrics
from ai.localization.model import LocalizationUNet


@pytest.fixture
def localization_tree(tmp_path):
    root = tmp_path / "localization_dataset"
    fields = ("split", "source_folder", "output_image_path", "output_mask_path")
    rows = []
    for split, folder in (("train", "folder_train"), ("val", "folder_val"), ("test", "folder_test")):
        for category in ("images", "masks"):
            (root / split / category).mkdir(parents=True)
        rgb = np.zeros((256, 256, 3), np.uint8)
        rgb[32:96, 40:120] = (180, 90, 20)
        mask = np.zeros((256, 256), np.uint8)
        mask[32:96, 40:120] = 255
        Image.fromarray(rgb, mode="RGB").save(root / split / "images" / "sample.png")
        Image.fromarray(mask, mode="L").save(root / split / "masks" / "sample.png")
        rows.append({"split": split, "source_folder": folder,
                     "output_image_path": f"{split}/images/sample.png",
                     "output_mask_path": f"{split}/masks/sample.png"})
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return root


def test_dataset_loading_shapes_binary_and_pairing(localization_tree):
    ds = LocalizationDataset(localization_tree, "train")
    image, mask = ds[0]
    assert image.shape == (3, 256, 256) and image.dtype == torch.float32
    assert mask.shape == (1, 256, 256) and mask.dtype == torch.float32
    assert set(torch.unique(mask).tolist()) == {0.0, 1.0}
    assert len(ds) == 1


def test_loader_rejects_missing_counterpart(localization_tree):
    (localization_tree / "train" / "masks" / "sample.png").unlink()
    with pytest.raises(ValueError, match="Unpaired"):
        LocalizationDataset(localization_tree, "train")


def test_model_forward_shape_and_logits():
    model = LocalizationUNet(base_channels=4).eval()
    with torch.no_grad():
        model.head.bias.fill_(2.0)
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 256, 256))
    assert output.shape == (1, 1, 256, 256)
    assert output.min() < 0 or output.max() > 1


def test_bce_dice_loss_finite():
    loss = BCEDiceLoss()(torch.randn(2, 1, 32, 32), torch.zeros(2, 1, 32, 32))
    assert torch.isfinite(loss)


def test_segmentation_metric_perfect_prediction():
    target = torch.zeros(1, 1, 8, 8)
    target[..., 2:5, 2:5] = 1
    result = segmentation_metrics(target, target)
    assert result["dice"] == pytest.approx(1.0)
    assert result["iou"] == pytest.approx(1.0)
    assert result["pixel_precision"] == pytest.approx(1.0)
    assert result["pixel_recall"] == pytest.approx(1.0)
    assert result["pixel_f1"] == pytest.approx(1.0)
    matching_logits = torch.where(target > 0, 10.0, -10.0)
    assert segmentation_metrics(matching_logits, target, from_logits=True)["dice"] == pytest.approx(1.0)


def test_segmentation_metric_precision_recall_f1():
    truth = torch.tensor([[[[1., 1., 0., 0.]]]])
    pred = torch.tensor([[[[1., 0., 1., 0.]]]])
    result = segmentation_metrics(pred, truth)
    assert result["pixel_precision"] == pytest.approx(0.5)
    assert result["pixel_recall"] == pytest.approx(0.5)
    assert result["pixel_f1"] == pytest.approx(0.5)
    assert result["iou"] == pytest.approx(1 / 3)


def test_checkpoint_roundtrip_and_inference(localization_tree, tmp_path):
    model = LocalizationUNet(base_channels=4)
    checkpoint = tmp_path / "localization.pt"
    torch.save({"model_state_dict": model.state_dict(), "epoch": 1,
                "config": {"base_channels": 4}}, checkpoint)
    loaded, payload = load_localization_checkpoint(checkpoint)
    assert payload["epoch"] == 1
    prob, binary = predict_image(loaded, localization_tree / "test" / "images" / "sample.png")
    assert prob.shape == binary.shape == (256, 256)
    assert np.isfinite(prob).all() and prob.min() >= 0 and prob.max() <= 1
    assert set(np.unique(binary).tolist()) <= {0, 1}
    saved = save_binary_mask(binary, tmp_path / "prediction.png")
    with Image.open(saved) as mask:
        assert mask.mode == "L" and mask.size == (256, 256)
