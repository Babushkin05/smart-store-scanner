"""Inference engine via tbn-runtime.

Loads ONNX model, preprocesses images, runs ternary/binary inference.
"""

import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ImageNet normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class InferenceEngine:
    """tbn-runtime wrapper for vegetable classification."""

    def __init__(self, model_path: str, labels_path: str,
                 use_quantization: bool = True):
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.use_quantization = use_quantization
        self._labels = self._load_labels()
        self._model = None

    def _load_labels(self) -> list:
        with open(self.labels_path) as f:
            return [line.strip() for line in f if line.strip()]

    def load(self):
        """Load ONNX model via tbn."""
        import tbn
        self._model = tbn.load_model(str(self.model_path))
        logger.info(f'Model loaded: {len(self._labels)} classes')

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """RGB [H,W,3] → normalized [1,3,224,224] float32 tensor."""
        import cv2

        img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        img = np.transpose(img, (2, 0, 1))        # HWC → CHW
        img = np.expand_dims(img, axis=0)          # add batch
        return img.astype(np.float32)

    def predict(self, image: np.ndarray) -> dict:
        """Run inference, return {class_name, confidence, inference_ms}."""
        if self._model is None:
            self.load()

        tensor = self.preprocess(image)

        start = time.perf_counter()
        if self.use_quantization:
            output = self._model.run_quantized(tensor)
        else:
            output = self._model.run(tensor)
        elapsed = time.perf_counter() - start

        probs = self._softmax(output[0])
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])

        return {
            'class_name': self._labels[idx],
            'confidence': round(confidence, 4),
            'inference_ms': round(elapsed * 1000, 1),
        }

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()
