"""Smart Store Scanner — Raspberry Pi Client.

Flask app:
- GET  /        → web UI
- POST /scan    → capture + tbn inference → result + price
- GET  /status  → health check (demo/live mode)
"""

import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, render_template, send_file

sys.path.insert(0, str(Path(__file__).parent))

from scanner.camera import Camera
from scanner.inference import InferenceEngine

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger('scanner')

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'model'

app = Flask(__name__)

# Config (env vars)
MODEL_PATH = os.environ.get('TBN_MODEL_PATH', str(MODEL_DIR / 'fruits_model.onnx'))
LABELS_PATH = os.environ.get('TBN_LABELS_PATH', str(MODEL_DIR / 'labels.txt'))
SERVER_URL = os.environ.get('SERVER_URL', 'http://127.0.0.1:8080')
USE_QUANTIZATION = os.environ.get('TBN_QUANTIZATION', '0') == '1'

# Lazy init
_camera: Camera | None = None
_engine: InferenceEngine | None = None
_last_result: dict = {}

# Built-in price map (used when Go server unavailable)
PRICES = {
    'Bean': 50, 'Bitter_Gourd': 70, 'Bottle_Gourd': 65, 'Brinjal': 55,
    'Broccoli': 120, 'Cabbage': 60, 'Capsicum': 90, 'Carrot': 45,
    'Cauliflower': 100, 'Cucumber': 55, 'Papaya': 130, 'Potato': 40,
    'Pumpkin': 80, 'Radish': 35, 'Tomato': 70,
}
EMOJIS = {
    'Bean': '\U0001FAD8', 'Bitter_Gourd': '\U0001F952', 'Bottle_Gourd': '\U0001FAD7',
    'Brinjal': '\U0001F346', 'Broccoli': '\U0001F966', 'Cabbage': '\U0001F96C',
    'Capsicum': '\U0001FAD1', 'Carrot': '\U0001F955', 'Cauliflower': '\U0001F966',
    'Cucumber': '\U0001F952', 'Papaya': '\U0001FAD3', 'Potato': '\U0001F954',
    'Pumpkin': '\U0001F383', 'Radish': '\U0001FADC', 'Tomato': '\U0001F345',
}


def get_camera() -> Camera:
    global _camera
    if _camera is None:
        _camera = Camera()
    return _camera


def get_engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        _engine = InferenceEngine(
            model_path=MODEL_PATH,
            labels_path=LABELS_PATH,
            use_quantization=USE_QUANTIZATION,
        )
    return _engine


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/status')
def status():
    eng = get_engine()
    return jsonify({
        'mode': 'demo' if eng.is_demo else 'live',
        'tbn_available': eng.is_demo is False or False,  # actual tbn status
        'classes': len(eng._labels),
        'camera_backend': get_camera()._backend,
    })


@app.route('/scan', methods=['POST'])
def scan():
    global _last_result

    try:
        # 1. Capture
        cam = get_camera()
        image = cam.capture()
        logger.info(f'Captured: {image.shape}')

        # Save for preview
        _capture_path = BASE_DIR / 'static' / 'capture.jpg'
        _capture_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(_capture_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # 2. Infer
        eng = get_engine()
        result = eng.predict(image)

        # 3. Assign local price (overridden by Go server if available)
        cls = result['class_name']
        result['emoji'] = EMOJIS.get(cls, '\U0001F4E6')
        result['price'] = PRICES.get(cls, 99)

        # 4. Try Go server
        try:
            resp = requests.post(
                f'{SERVER_URL}/scan',
                json={'class_name': cls, 'confidence': result['confidence']},
                timeout=2,
            )
            if resp.status_code == 200:
                srv = resp.json()
                result['price'] = srv.get('price', result['price'])
                result['cart_total'] = srv.get('cart_total')
                result['cart_items'] = srv.get('cart_items')
        except (requests.ConnectionError, requests.Timeout):
            pass  # use local price

        _last_result = result
        return jsonify(result)

    except Exception as e:
        logger.exception('Scan failed')
        return jsonify({'error': str(e)}), 500


@app.route('/last')
def last():
    return jsonify(_last_result or {})


@app.route('/capture.jpg')
def capture_preview():
    """Serve the last captured frame so the UI can show it."""
    capture_path = BASE_DIR / 'static' / 'capture.jpg'
    if capture_path.exists():
        return send_file(str(capture_path), mimetype='image/jpeg')
    return '', 404


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    eng = get_engine()
    eng.load()  # pre-load model + warmup at startup, not first scan
    cam = get_camera()  # init camera early

    logger.info('=' * 50)
    logger.info('Smart Store Scanner')
    logger.info(f'  Mode:      {"DEMO" if eng.is_demo else "LIVE"}')
    logger.info(f'  Model:     {MODEL_PATH}')
    logger.info(f'  Labels:    {len(eng._labels)} classes')
    logger.info(f'  Server:    {SERVER_URL}')
    logger.info(f'  Camera:    {cam._backend}')
    logger.info('=' * 50)

    app.run(host='0.0.0.0', port=5000, debug=False)
