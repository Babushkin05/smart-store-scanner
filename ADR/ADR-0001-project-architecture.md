# ADR-0001: Overall Project Architecture

**Date:** 2026-06-16
**Status:** Accepted

## Context

Building a Smart Store Scanner — an IoT product recognition system for a small
store. A Raspberry Pi with a camera captures product images, runs on-device
inference, and displays the result (product name + price) on a screen.

The inference must use the `tbn-runtime` library (fast-arm-tbcnn) which
accelerates Conv2D/Gemm on ARM via ternary/binary weight quantization.

## Decision

**Single-device architecture: everything runs on the Raspberry Pi.**

```
┌─────────────────────────────────────────┐
│ Raspberry Pi                             │
│  ┌──────────┐  ┌──────────┐  ┌───────┐ │
│  │ Camera   │→ │ tbn      │→ │ Flask │  │
│  │ (picam2) │  │ inference│  │ web   │  │
│  └──────────┘  └──────────┘  │ server│  │
│                               └───┬───┘ │
│                                   │     │
│                          ┌────────▼───┐ │
│                          │ Browser    │ │
│                          │ (kiosk)    │ │
│                          └────────────┘ │
└─────────────────────────────────────────┘
```

Components:
1. **Camera module** — captures frames via OpenCV (compatible with Pi Camera
   Module 3, USB webcams, or any V4L2 device).
2. **Inference engine** — wraps `tbn-runtime` Python bindings. Loads a
   quantized ONNX model, runs inference in <100ms on Cortex-A72.
3. **Flask web server** — local HTTP server (127.0.0.1:5000). Two endpoints:
   - `POST /scan` — triggers capture + inference, returns JSON with prediction.
   - `GET /` — serves the dashboard UI.
4. **Browser UI** — Chromium in kiosk mode on the Pi display. Shows product
   card (emoji, name, price) and scan history.

## Alternatives Considered

### A) Pi → Server architecture
Pi sends images to a cloud/server for inference. Rejected — adds network
dependency, latency, and contradicts the purpose of on-device tbn-runtime.

### B) Tkinter/PyGame GUI instead of web
Tkinter is simpler but harder to style. A web UI with HTML/CSS is more
flexible, supports touchscreens easily, and allows remote access if needed.

### C) No display — just send to server
User explicitly asked for on-screen display with product + price.

## Consequences

- **Positive:** Self-contained, works offline, low latency.
- **Positive:** Web UI can be accessed from a phone for demos.
- **Negative:** Pi must run both inference and a browser (modest load).
- **Negative:** Need to build tbn-runtime from source on the Pi (one-time setup).
