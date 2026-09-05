"""Pluggable face detection + embedding engines."""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from ai.face_verification.config import FaceVerificationConfig
from ai.face_verification.images import crop_with_padding
from ai.face_verification.quality import assess_face_crop
from ai.face_verification.types import BoundingBox, DetectedFace

logger = logging.getLogger("maya.ai.face_verification.engine")

_ENGINE_CACHE: dict[str, FaceEngine] = {}


class FaceEngine(Protocol):
    name: str
    model_name: str
    model_version: str

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        """Return all detected faces (may be empty)."""

    def embed(self, face_crop: np.ndarray) -> np.ndarray:
        """Return an L2-friendly identity embedding for a cropped face."""


class OpenCvHaarEngine:
    """OpenCV Haar detection + compact RGB-grid embedding (no extra downloads).

    Used when ``facenet-pytorch`` is unavailable. Identity quality is weaker than
    FaceNet; MATCH / NO_MATCH still use the same threshold contract.
    """

    name = "opencv_haar"

    def __init__(self, config: FaceVerificationConfig) -> None:
        import cv2

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError("OpenCV Haar cascade failed to load")
        self._cv2 = cv2
        self._config = config
        self.model_name = "opencv_haar_lbpgrid"
        self.model_version = "opencv-haar-1"

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        gray = self._cv2.cvtColor(image, self._cv2.COLOR_RGB2GRAY)
        rects = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self._config.min_face_size, self._config.min_face_size),
        )
        faces: list[DetectedFace] = []
        for x, y, w, h in rects:
            bbox = BoundingBox(int(x), int(y), int(x + w), int(y + h), score=1.0)
            crop = crop_with_padding(
                image, bbox.x1, bbox.y1, bbox.x2, bbox.y2, self._config.crop_padding
            )
            quality = assess_face_crop(crop, self._config)
            faces.append(DetectedFace(bbox=bbox, crop=crop, quality=quality))
        return faces

    def embed(self, face_crop: np.ndarray) -> np.ndarray:
        resized = self._cv2.resize(
            face_crop,
            (self._config.embed_size, self._config.embed_size),
            interpolation=self._cv2.INTER_AREA,
        )
        # 8x8 mean grid (RGB) → 192-d descriptor; no network weights required.
        cells = 8
        step = self._config.embed_size // cells
        parts: list[np.ndarray] = []
        for row in range(cells):
            for col in range(cells):
                block = resized[row * step : (row + 1) * step, col * step : (col + 1) * step]
                parts.append(block.reshape(-1, 3).mean(axis=0))
        vector = np.concatenate(parts).astype(np.float64)
        return vector


class FacenetPytorchEngine:
    """MTCNN detection + InceptionResnetV1 embeddings (vggface2 / casia-webface)."""

    name = "facenet_pytorch"

    def __init__(self, config: FaceVerificationConfig) -> None:
        try:
            from facenet_pytorch import InceptionResnetV1, MTCNN
        except ImportError as exc:
            raise ImportError(
                "facenet-pytorch is required for the facenet_pytorch engine"
            ) from exc

        import torch

        if config.cache_dir is not None:
            config.cache_dir.mkdir(parents=True, exist_ok=True)

        device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self._device = device
        self._torch = torch
        self._config = config
        self._mtcnn = MTCNN(
            image_size=config.embed_size,
            keep_all=True,
            device=device,
            post_process=True,
        )
        pretrained = config.pretrained or "vggface2"
        self._resnet = InceptionResnetV1(pretrained=pretrained).eval().to(device)
        self.model_name = config.model_name or "inception_resnet_v1"
        self.model_version = config.model_version or pretrained

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        from PIL import Image

        pil = Image.fromarray(image)
        boxes, probs = self._mtcnn.detect(pil)
        if boxes is None:
            return []
        scores = probs if probs is not None else [1.0] * len(boxes)
        faces: list[DetectedFace] = []
        for box, prob in zip(boxes, scores):
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            bbox = BoundingBox(x1, y1, x2, y2, score=float(prob) if prob is not None else 1.0)
            crop = crop_with_padding(
                image, bbox.x1, bbox.y1, bbox.x2, bbox.y2, self._config.crop_padding
            )
            if crop.size == 0:
                continue
            quality = assess_face_crop(crop, self._config)
            faces.append(DetectedFace(bbox=bbox, crop=crop, quality=quality))
        return faces

    def embed(self, face_crop: np.ndarray) -> np.ndarray:
        from PIL import Image
        from torchvision.transforms import functional as TF

        pil = Image.fromarray(face_crop).convert("RGB").resize(
            (self._config.embed_size, self._config.embed_size)
        )
        tensor = TF.to_tensor(pil)
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.unsqueeze(0).to(self._device)
        with self._torch.no_grad():
            embedding = self._resnet(tensor)
        return embedding.squeeze(0).detach().cpu().numpy().astype(np.float64)


def create_face_engine(config: FaceVerificationConfig) -> FaceEngine:
    engine = (config.engine or "opencv").lower()
    if engine in {"facenet_pytorch", "facenet", "mtcnn"}:
        try:
            return FacenetPytorchEngine(config)
        except ImportError:
            logger.warning(
                "facenet-pytorch is not installed; falling back to OpenCV Haar engine"
            )
            return OpenCvHaarEngine(config)
    if engine in {"opencv", "opencv_haar", "haar"}:
        return OpenCvHaarEngine(config)
    raise ValueError(f"Unknown face verification engine '{config.engine}'")


def get_face_engine(config: FaceVerificationConfig | None = None) -> FaceEngine:
    cfg = config or FaceVerificationConfig()
    cache_key = f"{cfg.engine}:{cfg.pretrained}:{cfg.device}:{cfg.model_version}"
    cached = _ENGINE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    engine = create_face_engine(cfg)
    _ENGINE_CACHE[cache_key] = engine
    return engine


def reset_face_engine() -> None:
    _ENGINE_CACHE.clear()
