"""
单张/批量预测脚本。

加载训练好的 best_model.pth，对 test 目录图片进行推理，
输出预测结果与置信度。
"""

import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config import TEST_DIR, CHECKPOINT_DIR, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, DEVICE
from models.classifier import CatDogClassifier


def get_transform():
    """与验证集一致的预处理（不做增强）。"""
    return T.Compose([
        T.Resize(int(IMAGE_SIZE * 1.14)),
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def predict_image(model, image_path, transform, device):
    """对单张图片进行预测。"""
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = probabilities.max(1)

    label = "cat" if predicted.item() == 0 else "dog"
    return label, confidence.item()


def main():
    # 加载模型
    model = CatDogClassifier(num_classes=2, pretrained=False).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_DIR / "best_model.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    transform = get_transform()

    # 扫描测试集
    test_images = sorted(TEST_DIR.iterdir())
    results = []

    print(f"Found {len(test_images)} test images. Predicting...")

    for img_path in test_images:
        if img_path.is_dir():
            continue
        try:
            label, conf = predict_image(model, img_path, transform, DEVICE)
            results.append({
                "filename": img_path.name,
                "label": label,
                "confidence": round(conf, 4)
            })
            # print(f"{img_path.name}: {label} ({conf:.2%})")
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

    # 保存 CSV
    df = pd.DataFrame(results)
    csv_path = PROJECT_ROOT / "submission.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()