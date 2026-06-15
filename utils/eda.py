"""
数据探索分析（EDA）脚本。
一次性运行，输出数据集的统计报告与可视化图表。
"""

import os
import logging
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/eda_report.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


def analyze_dataset(data_dir: str) -> Dict:
    """
    扫描数据集目录，收集图片的尺寸、格式、完整性、类别分布等信息。

    Args:
        data_dir: 数据目录路径，如 'data/train'

    Returns:
        包含统计信息的字典
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    stats = {
        'total_files': 0,
        'valid_images': 0,
        'corrupted': 0,
        'widths': [],
        'heights': [],
        'formats': Counter(),
        'categories': Counter(),
    }

    logger.info(f"Start scanning: {root}")

    for file_path in root.iterdir():
        if file_path.is_dir():
            continue

        stats['total_files'] += 1
        suffix = file_path.suffix.lower()

        # 尝试打开图片
        try:
            with Image.open(file_path) as img:
                w, h = img.size
                fmt = img.format  # JPEG, PNG, etc.

                stats['valid_images'] += 1
                stats['widths'].append(w)
                stats['heights'].append(h)
                stats['formats'][fmt] += 1

                # 根据文件名前缀统计类别（仅 train 目录有效）
                name_lower = file_path.name.lower()
                if name_lower.startswith('cat'):
                    stats['categories']['cat'] += 1
                elif name_lower.startswith('dog'):
                    stats['categories']['dog'] += 1
                else:
                    stats['categories']['unknown'] += 1

        except Exception as e:
            stats['corrupted'] += 1
            logger.warning(f"Corrupted or unreadable: {file_path.name} | {e}")

    logger.info(f"Scan complete: {stats['valid_images']} valid, {stats['corrupted']} corrupted")
    return stats


def print_report(stats: Dict, data_dir: str) -> None:
    """
    打印并保存统计报告。
    """
    widths = np.array(stats['widths'])
    heights = np.array(stats['heights'])

    report_lines = [
        "=" * 50,
        f"Dataset EDA Report: {data_dir}",
        "=" * 50,
        f"Total files scanned     : {stats['total_files']}",
        f"Valid images            : {stats['valid_images']}",
        f"Corrupted images        : {stats['corrupted']}",
        "",
        "--- Dimension Statistics ---",
        f"Width  -> min: {widths.min():5d}, max: {widths.max():5d}, mean: {widths.mean():.1f}, median: {int(np.median(widths))}",
        f"Height -> min: {heights.min():5d}, max: {heights.max():5d}, mean: {heights.mean():.1f}, median: {int(np.median(heights))}",
        "",
        "--- Format Distribution ---",
    ]
    for fmt, count in stats['formats'].items():
        report_lines.append(f"  {fmt}: {count}")

    report_lines.append("")
    report_lines.append("--- Category Distribution ---")
    for cat, count in stats['categories'].items():
        report_lines.append(f"  {cat}: {count}")

    report_lines.append("=" * 50)

    report_text = "\n".join(report_lines)
    print(report_text)

    # 同时保存到文本文件
    with open('logs/eda_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)


def visualize(stats: Dict, save_dir: str = 'logs') -> None:
    """
    生成可视化图表：尺寸分布直方图、宽高比散点图、类别分布图。
    """
    widths = np.array(stats['widths'])
    heights = np.array(stats['heights'])
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 宽度分布
    axes[0, 0].hist(widths, bins=50, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Width Distribution')
    axes[0, 0].set_xlabel('Width (pixels)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].axvline(224, color='red', linestyle='--', label='target=224')
    axes[0, 0].legend()

    # 2. 高度分布
    axes[0, 1].hist(heights, bins=50, color='salmon', edgecolor='black')
    axes[0, 1].set_title('Height Distribution')
    axes[0, 1].set_xlabel('Height (pixels)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].axvline(224, color='red', linestyle='--', label='target=224')
    axes[0, 1].legend()

    # 3. 宽高比散点图
    ratios = widths / heights
    axes[1, 0].scatter(widths, heights, c=ratios, cmap='viridis', alpha=0.5, s=10)
    axes[1, 0].set_title('Width vs Height (color = aspect ratio)')
    axes[1, 0].set_xlabel('Width')
    axes[1, 0].set_ylabel('Height')
    axes[1, 0].axvline(224, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].axhline(224, color='red', linestyle='--', alpha=0.5)

    # 4. 类别分布（仅 train 有数据）
    categories = stats['categories']
    if categories:
        axes[1, 1].bar(categories.keys(), categories.values(), color=['#3498db', '#e74c3c', '#95a5a6'])
        axes[1, 1].set_title('Category Distribution')

    plt.tight_layout()
    save_file = save_path / 'eda_visualization.png'
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    logger.info(f"Visualization saved to: {save_file}")
    plt.show()


if __name__ == '__main__':
    # 确保 logs 目录存在
    Path('logs').mkdir(exist_ok=True)

    # 分析训练集
    try:
        train_stats = analyze_dataset('./data/train')
        print_report(train_stats, './data/train')
        visualize(train_stats)
    except Exception as e:
        logger.exception("EDA failed")
        raise