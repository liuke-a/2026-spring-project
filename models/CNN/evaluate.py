"""
SimpleCNN 模型评估脚本。

在验证集上计算 Accuracy、Precision、Recall、F1，并绘制混淆矩阵。
"""

import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

# 将项目根目录加入路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.CNN.config import (
    CNN_CHECKPOINT_DIR,
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from models.CNN.CNN import SimpleCNN
from utils.dataset import get_dataloaders


def evaluate():
    """加载最佳模型并在验证集上评估。"""
    # 加载模型
    model = SimpleCNN(num_classes=2).to(DEVICE)
    checkpoint_path = CNN_CHECKPOINT_DIR / "best_model.pth"

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please train the model first: python models/CNN/train.py")
        return

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, "
          f"best_acc={checkpoint['best_acc']:.4f}")

    # 获取验证集
    _, val_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

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
    print("SimpleCNN Validation Set Evaluation")
    print("=" * 40)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("=" * 40)

    # 混淆矩阵
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
    plt.title("SimpleCNN Confusion Matrix")
    plt.tight_layout()

    output_path = PROJECT_ROOT / "logs" / "confusion_matrix_cnn.png"
    plt.savefig(output_path, dpi=150)
    print(f"Confusion matrix saved to: {output_path}")


if __name__ == "__main__":
    evaluate()
