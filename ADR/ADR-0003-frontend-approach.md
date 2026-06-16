# ADR-0003: Frontend and UI Approach

**Date:** 2026-06-16
**Status:** Accepted

## Context

The system needs a simple, visually clear display for a store environment.
When a product is scanned, the screen shows:
- Product picture/emoji
- Product name
- A fictional price
- Scan history

The UI runs on a touchscreen attached to the Raspberry Pi (or any HDMI display
with a mouse).

## Decision

**Flask + vanilla HTML/CSS/JS, displayed in Chromium kiosk mode.**

```
┌──────────────────────────────────────┐
│           Smart Store Scanner        │
│                                      │
│         ┌──────────────────┐         │
│         │                  │         │
│         │   🍎             │         │
│         │                  │         │
│         │   Apple          │         │
│         │   120 ₽          │         │
│         │                  │         │
│         └──────────────────┘         │
│                                      │
│   [ Scan Next Product ]             │
│                                      │
│   ── History ────────────────────── │
│   🍎 Apple ......... 120 ₽  (2m ago)│
│   🍋 Lemon ......... 85 ₽  (5m ago) │
│   🍅 Tomato ........ 60 ₽  (8m ago) │
└──────────────────────────────────────┘
```

### Technical details

1. **Flask** serves two routes:
   - `GET /` — renders the dashboard HTML.
   - `POST /scan` — triggers camera capture + inference, returns JSON:
     `{"product": "apple", "price": 120, "confidence": 0.92, "emoji": "🍎"}`.

2. **Frontend** uses vanilla JS (no framework) for simplicity:
   - `fetch('/scan', {method: 'POST'})` triggers a scan.
   - Result is rendered as a product card with CSS animation.
   - History is stored in a JS array (in-memory, resets on page reload).
   - Auto-refresh every 10 seconds (optional).

3. **Kiosk mode** on Raspberry Pi:
   ```bash
   chromium-browser --kiosk --incognito http://127.0.0.1:5000
   ```

## Product catalog (hardcoded)

| Class | Emoji | Price (₽) |
|-------|-------|-----------|
| apple | 🍎 | 120 |
| banana | 🍌 | 90 |
| bell pepper | 🫑 | 150 |
| carrot | 🥕 | 60 |
| cucumber | 🥒 | 80 |
| lemon | 🍋 | 85 |
| orange | 🍊 | 110 |
| tomato | 🍅 | 70 |

## Alternatives Considered

### A) Tkinter GUI
Simpler deployment (no browser needed) but limited styling. Rejected — HTML/CSS
gives a much more polished look for a store demo.

### B) React/Vue SPA
Overkill for a single-page UI with one button. Vanilla JS is sufficient.

### C) PyGame with images
Good for full-screen graphics but harder to style text and layout. Web UI is
more maintainable.

## Consequences

- **Positive:** Clean, modern look with minimal code.
- **Positive:** Can be accessed from a phone on the same network for demos.
- **Positive:** CSS animations make the product reveal feel polished.
- **Negative:** Requires Chromium running on Pi (adds ~200 MB RAM usage).
- **Negative:** Not a true kiosk — needs manual browser launch or autostart config.
