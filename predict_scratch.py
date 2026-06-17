"""
从头训练模型（SE-ResNet）的预测脚本。

预测 SEResNet18 和 SEResNet34 需要调整 config_scratch.py 中的 MODEL_LAYERS 参数。

对 test 目录批量推理，输出预测标签与置信度到 submission_scratch.csv。
默认加载 checkpoints/scratch/best_model.pth 中的 EMA 权重。
"""

import sys
from pathlib import Path

import torch
from torch.amp import autocast
import torchvision.transforms as T
from PIL import Image
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config_scratch import (
    TEST_DIR, SCRATCH_CHECKPOINT_DIR, DEVICE,
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    MODEL_LAYERS, MODEL_WIDTH, DROP_RATE, DROP_PATH,
)
from models.seresnet import build_scratch_model


def get_transform():
    """与验证集一致的预处理（不做增强）。"""
    return T.Compose([
        T.Resize(int(IMAGE_SIZE * 1.14)),
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_model(use_ema: bool = True):
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


def predict_image(model, image_path, transform, device):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device).to(memory_format=torch.channels_last)

    with torch.no_grad():
        with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            output = model(input_tensor)
        probabilities = torch.softmax(output.float(), dim=1)
        confidence, predicted = probabilities.max(1)

    label = "cat" if predicted.item() == 0 else "dog"
    return label, float(confidence.item())


def main():
    model = load_model(use_ema=True)
    transform = get_transform()

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    test_images = sorted(
        p for p in TEST_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in valid_exts
    )

    results = []
    print(f"Found {len(test_images)} test images. Predicting...")

    for img_path in test_images:
        try:
            label, conf = predict_image(model, img_path, transform, DEVICE)
            results.append({
                "filename": img_path.name,
                "label": label,
                "confidence": round(conf, 4),
            })
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

    df = pd.DataFrame(results)
    csv_path = PROJECT_ROOT / "submission_scratch.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
