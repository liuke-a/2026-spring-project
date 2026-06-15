"""
模型评估脚本。

在验证集上计算 Accuracy、Precision、Recall、F1，并绘制混淆矩阵。
"""

import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config import CHECKPOINT_DIR, DEVICE, BATCH_SIZE, NUM_WORKERS, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from models.classifier import CatDogClassifier
from utils.dataset import get_dataloaders


def get_transform():
    return T.Compose([
        T.Resize(int(IMAGE_SIZE * 1.14)),
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def evaluate():
    # 加载模型
    model = CatDogClassifier(num_classes=2, pretrained=False).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_DIR / "best_model.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 获取验证集（从 train 划分的 10%）
    _, val_loader = get_dataloaders()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 计算指标
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="binary")
    recall = recall_score(all_labels, all_preds, average="binary")
    f1 = f1_score(all_labels, all_preds, average="binary")

    print("=" * 40)
    print("Validation Set Evaluation")
    print("=" * 40)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("=" * 40)

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["cat", "dog"], yticklabels=["cat", "dog"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("logs/confusion_matrix.png", dpi=150)
    print("Confusion matrix saved to: logs/confusion_matrix.png")


if __name__ == "__main__":
    evaluate()