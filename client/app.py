"""Smart Store Scanner — Raspberry Pi Client.

Flask app that:
- Serves a web UI with capture button + product display
- POST /scan  → captures image, runs tbn inference, optionally forwards to Go server
- GET /       → main UI
"""

import logging
import os
import sys
from pathlib import Path

import requests
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).parent))

from scanner.camera import Camera
from scanner.inference import InferenceEngine

# ── Setup ────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger('scanner')

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'model'

app = Flask(__name__)

# ── Config (env vars with defaults) ──────────────────────────────────────────

MODEL_PATH = os.environ.get('TBN_MODEL_PATH',
                            str(MODEL_DIR / 'fruits_model.onnx'))
LABELS_PATH = os.environ.get('TBN_LABELS_PATH',
                             str(MODEL_DIR / 'labels.txt'))
SERVER_URL = os.environ.get('SERVER_URL', 'http://127.0.0.1:8080')
USE_QUANTIZATION = os.environ.get('TBN_QUANTIZATION', '1') == '1'

# ── Init ─────────────────────────────────────────────────────────────────────

camera: Camera = None
engine: InferenceEngine = None
last_result: dict = {}


def get_camera() -> Camera:
    global camera
    if camera is None:
        camera = Camera()
    return camera


def get_engine() -> InferenceEngine:
    global engine
    if engine is None:
        engine = InferenceEngine(
            model_path=MODEL_PATH,
            labels_path=LABELS_PATH,
            use_quantization=USE_QUANTIZATION,
        )
    return engine


# ── Routes ───────────────────────────────────────────────────────────────────


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/scan', methods=['POST'])
def scan():
    """Capture image, run inference, forward to Go server, return result."""
    global last_result

    try:
        # 1. Capture
        cam = get_camera()
        image = cam.capture()
        logger.info(f'Captured: {image.shape}')

        # 2. Infer
        eng = get_engine()
        result = eng.predict(image)

        # 3. Forward to Go server (if available)
        server_result = {}
        try:
            resp = requests.post(
                f'{SERVER_URL}/scan',
                json={'class_name': result['class_name'],
                      'confidence': result['confidence']},
                timeout=2,
            )
            if resp.status_code == 200:
                server_result = resp.json()
        except (requests.ConnectionError, requests.Timeout):
            logger.warning('Go server unreachable, using local only')

        # 4. Combine
        result['price'] = server_result.get('price')
        result['cart_total'] = server_result.get('cart_total')
        result['cart_items'] = server_result.get('cart_items')
        last_result = result

        return jsonify(result)

    except Exception as e:
        logger.exception('Scan failed')
        return jsonify({'error': str(e)}), 500


@app.route('/last')
def last():
    """Return last scan result."""
    return jsonify(last_result or {})


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    logger.info('Starting Smart Store Scanner...')
    logger.info(f'  Model: {MODEL_PATH}')
    logger.info(f'  Server: {SERVER_URL}')
    logger.info(f'  Quantization: {USE_QUANTIZATION}')

    # Pre-load model
    try:
        get_engine().load()
    except Exception as e:
        logger.warning(f'Model not available: {e}')
        logger.warning('Scanner will run in demo mode (random predictions)')

    app.run(host='0.0.0.0', port=5000, debug=False)
