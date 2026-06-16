"""
从头训练模型（SE-ResNet）的评估脚本。

评估 SEResNet18 和 SEResNet34 需要调整 config_scratch.py 中的 MODEL_LAYERS 参数。
在验证集上计算 Accuracy / Precision / Recall / F1 并绘制混淆矩阵。
默认加载 checkpoints/scratch/best_model.pth 中的 EMA 权重。
"""

import sys
from pathlib import Path

import torch
from torch.amp import autocast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config_scratch import (
    SCRATCH_CHECKPOINT_DIR, DEVICE, BATCH_SIZE,
    MODEL_LAYERS, MODEL_WIDTH, DROP_RATE, DROP_PATH,
    MAX_SAMPLES, NUM_WORKERS,
)
from models.seresnet import build_scratch_model
from utils.dataset import get_dataloaders


def load_model(use_ema: bool = True):
    """构建 SE-ResNet 并加载 scratch 训练权重（默认用 EMA 权重）。"""
    ckpt_path = SCRATCH_CHECKPOINT_DIR / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = build_scratch_model(
        num_classes=2,
        layers=MODEL_LAYERS,
        width=MODEL_WIDTH,
        drop_rate=DROP_RATE,
        drop_path=DROP_PATH,
    ).to(DEVICE).to(memory_format=torch.channels_last)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    key = "ema_state_dict" if (use_ema and "ema_state_dict" in ckpt) else "model_state_dict"
    model.load_state_dict(ckpt[key])
    model.eval()
    return model


def evaluate():
    model = load_model(use_ema=True)

    # 与训练验证阶段保持尽可能一致
    _, val_loader = get_dataloaders(
        max_samples=MAX_SAMPLES,
        strong=False,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)
            labels = labels.to(DEVICE, non_blocking=True)

            with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                outputs = model(images)

            predicted = outputs.float().argmax(dim=1)
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="binary", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="binary", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)

    print("=" * 40)
    print("Validation Set Evaluation (SE-ResNet, scratch)")
    print("=" * 40)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("=" * 40)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["cat", "dog"],
        yticklabels=["cat", "dog"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (SE-ResNet scratch)")
    plt.tight_layout()

    out_dir = PROJECT_ROOT / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "confusion_matrix_scratch.png"
    plt.savefig(out_path, dpi=150)
    print(f"Confusion matrix saved to: {out_path}")


if __name__ == "__main__":
    evaluate()
