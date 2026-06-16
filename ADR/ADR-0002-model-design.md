# ADR-0002: Model Architecture and Training

**Date:** 2026-06-16
**Status:** Accepted

## Context

Need an image classification model for 15 vegetable classes: Bean, Bitter_Gourd,
Bottle_Gourd, Brinjal, Broccoli, Cabbage, Capsicum, Carrot, Cauliflower,
Cucumber, Papaya, Potato, Pumpkin, Radish, Tomato. The model must:
1. Be compatible with `tbn-runtime` operator set.
2. Be small enough for Raspberry Pi (target: <5 MB, <100ms inference).
3. Achieve reasonable accuracy on the limited classes.

`tbn-runtime` supports only: Conv, Relu, MaxPool, AveragePool, GlobalMaxPool,
GlobalAveragePool, Gemm/MatMul, Add, Sub, Reshape, Flatten, Greater, Less, Cast.

Critically unsupported: BatchNormalization, Softmax, Concat, Clip/ReLU6, Mul,
Pad, Transpose, Sigmoid.

## Decision

**Custom VGG-like CNN with 5 convolutional blocks, trained from scratch.**

Architecture:
```
Input [3, 224, 224]
  Block1: Conv(3→32, k3, pad1) → ReLU → MaxPool(2)    → [32, 112, 112]
  Block2: Conv(32→64, k3, pad1) → ReLU → MaxPool(2)   → [64, 56, 56]
  Block3: Conv(64→128, k3, pad1) → ReLU → MaxPool(2)  → [128, 28, 28]
  Block4: Conv(128→256, k3, pad1) → ReLU → MaxPool(2) → [256, 14, 14]
  Block5: Conv(256→512, k3, pad1) → ReLU              → [512, 14, 14]
  GlobalAveragePool → [512]
  Dropout(0.5) — inference only (evaluated mode = identity)
  Linear(512→15) → [15 классов]
```

All operators map directly to supported ONNX ops:
- `Conv2d` → ONNX Conv
- `ReLU` → ONNX Relu
- `MaxPool2d` → ONNX MaxPool
- `AdaptiveAvgPool2d(1)` → ONNX GlobalAveragePool
- `Flatten` → ONNX Flatten
- `Linear` → ONNX Gemm
- `Dropout` → removed during export (identity in eval mode)

No BatchNorm is used — the model relies on proper weight initialization
(Kaiming uniform) and a moderate learning rate.

## Training strategy

- **Dataset:** `misrakahmed/vegetable-image-dataset` (Kaggle) — 15 classes,
  21,000 images, 224×224, balanced (1000/200/200 per class per split).
- **Augmentation:** RandomHorizontalFlip, RandomRotation(±15°), ColorJitter.
- **Loss:** CrossEntropyLoss.
- **Optimizer:** Adam, lr=1e-3, ReduceLROnPlateau scheduler.
- **Epochs:** 25-30 (small model converges fast).
- **Output:** ONNX model with float32 weights.
- **Quantization:** tbn-runtime performs auto-quantization to ternary at
  inference time. No separate quantization step needed.

## Alternatives Considered

### A) Pre-trained MobileNetV2 + fine-tune
Rejected — tbn-runtime doesn't support BatchNorm or Clip ops used by MobileNet.

### B) Pre-trained model with BatchNorm fusion
Possible but fragile. Fusing BN into Conv weights requires post-processing the
ONNX graph. Custom model is simpler and guaranteed to work.

### C) Random weights (demo only)
Would show the pipeline but not actually classify. A trained model on 8 classes
can achieve >90% accuracy even with this simple architecture.

## Consequences

- **Positive:** Full control over architecture; guaranteed tbn-runtime compat.
- **Positive:** Small model (~2-3 MB ONNX, ~5 MB after tbn quantization).
- **Negative:** Requires training (GPU recommended, ~30 min on a laptop).
- **Negative:** Without BatchNorm, training needs more care with learning rate.
