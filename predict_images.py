"""
data/images 批量预测脚本。

使用 service/config.yaml 指定的模型，对 data/images/ 目录下所有图片进行推理，
输出预测结果到控制台并保存为 predict_results.csv。

Usage:
    python predict_images.py                     # 使用默认配置
    python predict_images.py --model seresnet34  # 切换模型
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from service import get_model_service


def main():
    parser = argparse.ArgumentParser(description="Predict images in data/images/")
    parser.add_argument(
        "--model", type=str, default=None,
        help="模型名（seresnet18 | seresnet34），默认从 service/config.yaml 读取",
    )
    args = parser.parse_args()

    # 获取模型服务
    print("=" * 50)
    svc = get_model_service(model=args.model)
    model_name = args.model or "seresnet18"  # 仅用于显示
    print(f"Model: {model_name}")
    print("=" * 50)

    # 扫描 data/images 目录
    image_dir = PROJECT_ROOT / "data" / "images"
    if not image_dir.exists():
        print(f"目录不存在: {image_dir}")
        return

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in valid_exts
    )

    if not image_paths:
        print(f"目录 {image_dir} 中没有找到图片文件。")
        return

    print(f"共找到 {len(image_paths)} 张图片，开始预测...\n")

    # 批量推理
    results = svc.predict_batch([str(p) for p in image_paths])

    # 打印结果
    rows = []
    for path, result in zip(image_paths, results):
        label = result["label"]
        conf = result["confidence"]
        cat_prob = result["probabilities"]["cat"]
        dog_prob = result["probabilities"]["dog"]
        print(f"  {path.name:<20s} → {label:<3s}  "
              f"cat={cat_prob:.4f}  dog={dog_prob:.4f}  "
              f"conf={conf:.4f}")
        rows.append({
            "filename": path.name,
            "label": label,
            "confidence": round(conf, 4),
            "prob_cat": round(cat_prob, 4),
            "prob_dog": round(dog_prob, 4),
        })

    # 保存 CSV
    df = pd.DataFrame(rows)
    csv_path = PROJECT_ROOT / "predict_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
