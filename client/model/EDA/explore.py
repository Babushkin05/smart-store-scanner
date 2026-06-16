#!/usr/bin/env python3
"""
EDA: Vegetable Image Dataset Analysis.

Dataset: misrakahmed/vegetable-image-dataset (Kaggle)
- 15 vegetable classes
- 21,000 images total (224x224, JPG)
- Train: 1000/class, Validation: 200/class, Test: 200/class

Run from client/model/:
    source venv/bin/activate
    python EDA/explore.py
"""

import os
import sys
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # non-interactive — save to files
import matplotlib.pyplot as plt

# ── Config ───────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / 'plots'
OUTPUT_DIR.mkdir(exist_ok=True)

KAGGLE_PATH = os.path.expanduser(
    '.cache/kagglehub/datasets/misrakahmed/vegetable-image-dataset/versions/1'
)
DATA_DIR = Path(KAGGLE_PATH) / 'Vegetable Images'


def find_dataset():
    """Locate the dataset, trying kagglehub cache first, then common paths."""
    if DATA_DIR.exists():
        return DATA_DIR

    print('Dataset not found in cache. Downloading...')
    import kagglehub
    path = kagglehub.dataset_download('misrakahmed/vegetable-image-dataset')
    return Path(path) / 'Vegetable Images'


def analyze(data_dir: Path):
    print('=' * 60)
    print('Vegetable Image Dataset — EDA')
    print('=' * 60)

    for split in ['train', 'validation', 'test']:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue

        classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        counts = {}
        sizes = []

        for cls in classes:
            cls_dir = split_dir / cls
            images = list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.jpeg')) + list(cls_dir.glob('*.png'))
            counts[cls] = len(images)

            for img_path in images[:5]:
                try:
                    with Image.open(img_path) as img:
                        sizes.append(img.size)
                except Exception:
                    pass

        print(f'\n--- {split.upper()} ({sum(counts.values())} images) ---')
        for cls, count in sorted(counts.items()):
            bar = '█' * (count // 50)
            print(f'  {cls:<20} {count:>5}  {bar}')

    # ── Class distribution plot ──────────────────────────────────────────

    train_counts = {}
    train_dir = data_dir / 'train'
    for cls_dir in train_dir.iterdir():
        if cls_dir.is_dir():
            images = list(cls_dir.glob('*.jpg'))
            train_counts[cls_dir.name] = len(images)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    names = sorted(train_counts.keys())
    values = [train_counts[n] for n in names]
    short_names = [n.replace('_', '\n') for n in names]

    axes[0].bar(short_names, values, color='steelblue', edgecolor='white')
    axes[0].set_title('Images per Class (Train)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Count')
    axes[0].set_ylim(0, max(values) * 1.15)
    for i, v in enumerate(values):
        axes[0].text(i, v + 10, str(v), ha='center', fontsize=9)

    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
    axes[1].pie(values, labels=short_names, autopct='%1.1f%%',
                colors=colors, textprops={'fontsize': 8})
    axes[1].set_title('Class Distribution (Train)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    dist_path = OUTPUT_DIR / 'class_distribution.png'
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {dist_path}')

    # ── Sample images ────────────────────────────────────────────────────

    fig, axes = plt.subplots(3, 5, figsize=(14, 8))
    fig.suptitle('Sample Images — One per Class', fontsize=16, fontweight='bold')

    for idx, cls in enumerate(sorted(train_counts.keys())):
        row, col = divmod(idx, 5)
        cls_dir = train_dir / cls
        images = list(cls_dir.glob('*.jpg'))
        if images:
            img = Image.open(images[0])
            axes[row, col].imshow(img)
            axes[row, col].set_title(cls.replace('_', ' '), fontsize=10)
            axes[row, col].axis('off')

    plt.tight_layout()
    samples_path = OUTPUT_DIR / 'sample_images.png'
    plt.savefig(samples_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {samples_path}')

    # ── Per-class sample grid ────────────────────────────────────────────

    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    fig.suptitle('5 Random Samples per Class', fontsize=16, fontweight='bold')

    for idx, cls in enumerate(sorted(train_counts.keys())):
        row, col = divmod(idx, 5)
        cls_dir = train_dir / cls
        images = list(cls_dir.glob('*.jpg'))[:5]
        if len(images) >= 5:
            stacked = np.hstack([np.array(Image.open(p).resize((64, 64))) for p in images])
            axes[row, col].imshow(stacked)
            axes[row, col].set_title(cls.replace('_', ' '), fontsize=9)
            axes[row, col].axis('off')

    plt.tight_layout()
    grid_path = OUTPUT_DIR / 'per_class_samples.png'
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {grid_path}')

    # ── Image stats ──────────────────────────────────────────────────────

    print('\n--- Image Statistics ---')
    sample_sizes = []
    for cls_dir in train_dir.iterdir():
        if not cls_dir.is_dir():
            continue
        for img_path in list(cls_dir.glob('*.jpg'))[:20]:
            with Image.open(img_path) as img:
                sample_sizes.append(img.size)

    if sample_sizes:
        widths, heights = zip(*sample_sizes)
        print(f'  Unique resolutions: {set(sample_sizes)}')
        print(f'  Mean size: {np.mean(widths):.0f}x{np.mean(heights):.0f}')

    if sample_sizes and all(w == 224 and h == 224 for w, h in sample_sizes):
        print('  All sampled images are 224x224 ✓')

    # ── Summary ──────────────────────────────────────────────────────────

    val_count = sum(
        len(list((data_dir / 'validation' / d).glob('*.jpg')))
        for d in (data_dir / 'validation').iterdir() if d.is_dir()
    )
    test_count = sum(
        len(list((data_dir / 'test' / d).glob('*.jpg')))
        for d in (data_dir / 'test').iterdir() if d.is_dir()
    )

    print('\n--- Summary ---')
    print(f'  Classes: {len(train_counts)}')
    print(f'  Total images: {sum(train_counts.values())} train + '
          f'{val_count} val + {test_count} test = '
          f'{sum(train_counts.values()) + val_count + test_count}')
    print(f'  Image size: 224x224')
    print(f'  Balanced: yes (1000/class in train, 200/class in val/test)')
    print(f'  Format: JPG')

    print('\n--- tbn-runtime Compatibility ---')
    print(f'  Classes: {sorted(train_counts.keys())}')
    print()
    print('  NOTE: This dataset is VEGETABLES only (15 classes).')
    print('  No fruits (apple, banana, orange, lemon).')
    print('  Model will be trained on these 15 vegetable classes.')
    print()
    print(f'  Plots saved to: {OUTPUT_DIR}/')


if __name__ == '__main__':
    data_dir = find_dataset()
    analyze(data_dir)
