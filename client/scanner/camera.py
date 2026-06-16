"""Camera capture for Raspberry Pi.

Supports Pi Camera Module 3 (picamera2) and USB webcams (OpenCV).
Auto-detects available backend.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# ── Backend detection ────────────────────────────────────────────────────────

HAS_CV2 = False
HAS_PICAMERA2 = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    pass

try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except ImportError:
    pass


class Camera:
    """Unified camera interface."""

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self._backend = None
        self._cap = None
        self._fallback_img = None

        if HAS_PICAMERA2:
            self._init_picamera2()
            self._backend = 'picamera2'
        elif HAS_CV2:
            self._init_cv2()
            self._backend = 'opencv'
        else:
            self._backend = 'fallback'
            logger.warning('No camera backend available, using test pattern')
            self._fallback_img = self._make_fallback()

    def _init_picamera2(self):
        self._cap = Picamera2()
        config = self._cap.create_still_configuration(
            main={'size': (self.width, self.height)}
        )
        self._cap.configure(config)
        self._cap.start()
        logger.info('Camera: picamera2 ready')

    def _init_cv2(self):
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            raise RuntimeError('Cannot open camera 0')
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # Warm up
        for _ in range(5):
            self._cap.read()
        logger.info('Camera: OpenCV ready')

    def _make_fallback(self) -> np.ndarray:
        img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 128
        cv2 = __import__('cv2')
        cv2.putText(img, 'NO CAMERA', (100, self.height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        return img

    def capture(self) -> np.ndarray:
        """Capture a frame, return RGB numpy array [H, W, 3]."""
        if self._backend == 'picamera2':
            frame = self._cap.capture_array()
            return frame  # RGB already

        elif self._backend == 'opencv':
            ret, frame = self._cap.read()
            if not ret:
                raise RuntimeError('Failed to capture frame')
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        else:
            return self._fallback_img.copy()

    def close(self):
        if self._backend == 'picamera2':
            self._cap.stop()
        elif self._backend == 'opencv':
            self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
