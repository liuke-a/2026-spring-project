"""
SimpleCNN 预测脚本。

加载训练好的 best_model.pth，对 test 目录图片进行推理，
输出预测结果并保存为 submission_cnn.csv。
"""

import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
import pandas as pd

# 将项目根目录加入路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.CNN.config import (
    TEST_DIR,
    CNN_CHECKPOINT_DIR,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    DEVICE,
)
from models.CNN.CNN import SimpleCNN


def get_transform():
    """与验证集一致的预处理（不做增强）。"""
    return T.Compose([
        T.Resize(int(IMAGE_SIZE * 1.14)),
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def predict_image(model, image_path, transform, device):
    """对单张图片进行预测，返回 (label, confidence, prob_dog)。"""
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = probabilities.max(1)

    label = "cat" if predicted.item() == 0 else "dog"
    prob_dog = probabilities[0, 1].item()
    return label, confidence.item(), prob_dog


def main():
    # 加载模型
    checkpoint_path = CNN_CHECKPOINT_DIR / "best_model.pth"
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please train the model first: python models/CNN/train.py")
        return

    model = SimpleCNN(num_classes=2).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, "
          f"best_acc={checkpoint['best_acc']:.4f}")

    transform = get_transform()

    # 扫描测试集
    test_images = sorted(TEST_DIR.iterdir())
    results = []

    print(f"Found {len(test_images)} test images. Predicting...")

    for img_path in test_images:
        if img_path.is_dir():
            continue
        try:
            label, conf, prob_dog = predict_image(model, img_path, transform, DEVICE)
            results.append({
                "filename": img_path.name,
                "label": label,
                "confidence": round(conf, 4),
            })
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

    # 保存 CSV（与 predict.py 相同格式）
    df = pd.DataFrame(results)
    csv_path = PROJECT_ROOT / "submission_cnn.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    print(f"Total: {len(df)} predictions")
    print(f"  cat: {(df['label'] == 'cat').sum()}, "
          f"dog: {(df['label'] == 'dog').sum()}")


if __name__ == "__main__":
    main()
