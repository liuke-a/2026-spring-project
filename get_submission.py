"""
生成竞赛提交文件。

使用 service/config.yaml 指定的模型对 data/test/ 目录下全部图片进行批量推理，
输出 submission.csv，列为 id 与 label（狗的概率，浮点数）。

Usage:
    python get_submission.py                     # 默认模型（seresnet18）
    python get_submission.py --model seresnet34  # 切换模型
    python get_submission.py -o submission.csv   # 指定输出路径
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config import TEST_DIR
from service import get_model_service


def extract_id(filename: str) -> int:
    """从文件名提取数字 ID，如 '123.jpg' -> 123。"""
    return int(Path(filename).stem)


def main():
    parser = argparse.ArgumentParser(
        description="Generate submission CSV for data/test/"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="模型名（seresnet18 | seresnet34），默认从 service/config.yaml 读取",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="submission.csv",
        help="输出 CSV 路径，默认 submission.csv",
    )
    args = parser.parse_args()

    # 获取模型
    print("=" * 50)
    svc = get_model_service(model=args.model)
    print(f"Model loaded, predicting {TEST_DIR} ...")
    print("=" * 50)

    # 扫描测试集（按数字 ID 排序，保证输出顺序稳定）
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    test_images = sorted(
        p for p in TEST_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in valid_exts
    )

    if not test_images:
        print(f"No images found in {TEST_DIR}")
        return

    print(f"Found {len(test_images)} test images. Predicting...\n")

    # 批量推理（BaseModelService.predict_batch 自动分片防 OOM）
    results = svc.predict_batch([str(p) for p in test_images])

    # 构建提交格式
    rows = []
    for path, result in zip(test_images, results):
        rows.append({
            "id": extract_id(path.name),
            "label": result["probabilities"]["dog"],  # 图片为狗的概率
        })

    # 按 id 排序（match sampleSubmission 格式）
    rows.sort(key=lambda r: r["id"])

    df = pd.DataFrame(rows, columns=["id", "label"])
    output_path = PROJECT_ROOT / args.output
    df.to_csv(output_path, index=False)

    print(f"Submission saved to: {output_path}")
    print(f"Total: {len(df)} predictions")
    cat_count = (df["label"] == 0).sum()
    dog_count = (df["label"] == 1).sum()
    print(f"  cat (0): {cat_count}, dog (1): {dog_count}")


if __name__ == "__main__":
    main()
