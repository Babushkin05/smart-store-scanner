"""Inference engine via tbn-runtime.

Loads ONNX model, preprocesses images, runs ternary/binary inference.
Falls back to random predictions (demo mode) when tbn is not available.
"""

import logging
import random
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ImageNet normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── tbn availability ─────────────────────────────────────────────────────────

HAS_TBN = False

# Auto-detect tbn from sibling fast-arm-tbcnn build directory
_self = Path(__file__).resolve()
_repo_root = _self.parent.parent.parent           # smart-store-scanner/
_docs_dir = _repo_root.parent                      # docs/ (on Pi) or etc/ (on Mac)
_tbn_candidates = [
    _docs_dir / 'fast-arm-tbcnn' / 'build' / 'python',
    _docs_dir / 'fast-arm-tbcnn' / 'tbn-runtime' / 'build' / 'python',
    Path('/home/pi/docs/fast-arm-tbcnn/build/python'),
]
for _p in _tbn_candidates:
    if _p.exists():
        sys.path.insert(0, str(_p))

try:
    import tbn  # noqa: F401
    HAS_TBN = True
except ImportError:
    pass


class InferenceEngine:
    """tbn-runtime wrapper. Falls back to demo mode if tbn not installed."""

    def __init__(self, model_path: str = '', labels_path: str = '',
                 use_quantization: bool = True):
        self.model_path = Path(model_path) if model_path else None
        self.labels_path = Path(labels_path) if labels_path else None
        self.use_quantization = use_quantization
        self._labels = self._load_labels()
        self._model = None
        self._demo_mode = not HAS_TBN or not model_path or not Path(model_path).exists()

        if self._demo_mode:
            logger.warning('Demo mode: using random predictions '
                           f'(tbn_available={HAS_TBN}, '
                           f'model_exists={Path(model_path).exists() if model_path else False})')
        else:
            logger.info(f'Live mode: {len(self._labels)} classes, '
                        f'quantization={use_quantization}')

    @property
    def is_demo(self) -> bool:
        return self._demo_mode

    def _load_labels(self) -> list:
        if self.labels_path and self.labels_path.exists():
            with open(self.labels_path) as f:
                return [line.strip() for line in f if line.strip()]
        # Fallback labels
        return ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli',
                'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber',
                'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']

    def load(self):
        """Load ONNX model via tbn (skipped in demo mode)."""
        if self._demo_mode:
            return
        # Use mmk=1, nmk=1 to support small batch/class dimensions
        self._model = tbn.load_model(str(self.model_path),
                                     mblk=64, nblk=64, kblk=128,
                                     mmk=1, nmk=1)
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
        """Run inference. Returns random result in demo mode."""
        start = time.perf_counter()

        if self._demo_mode:
            return self._predict_demo(start)

        if self._model is None:
            self.load()

        tensor = self.preprocess(image)

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
            'demo': False,
        }

    def _predict_demo(self, start_time: float) -> dict:
        """Return a random prediction for demo/testing."""
        idx = random.randint(0, len(self._labels) - 1)
        elapsed = time.perf_counter() - start_time
        return {
            'class_name': self._labels[idx],
            'confidence': round(random.uniform(0.70, 0.99), 4),
            'inference_ms': round(elapsed * 1000, 1),
            'demo': True,
        }

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()
