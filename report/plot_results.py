"""
从日志数据生成实验结果可视化图。
- 模型对比柱状图
- 训练过程热力图（预测分布变化）
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_model_comparison():
    """模型 Accuracy 与 F1-Score 对比柱状图。"""
    import numpy as np

    models = ["SE-ResNet18\n(11.7M params)", "SE-ResNet34\n(21.3M params)"]
    accuracy = [97.48, 98.56]
    f1_score = [97.45, 98.55]

    x = np.arange(len(models))
    width = 0.3

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, accuracy, width, label="Accuracy", color="#42A5F5")
    bars2 = ax.bar(x + width/2, f1_score, width, label="F1-Score", color="#FF7043")

    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(94, 100)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                f"{bar.get_height():.2f}%", ha="center", va="bottom", fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                f"{bar.get_height():.2f}%", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: model_comparison.png")


def plot_resource_comparison():
    """资源消耗对比图。"""
    models = ["SE-ResNet18", "SE-ResNet34"]
    params = [11.7, 21.3]
    gpu_mem = [4.4, 6.1]
    train_time = [35, 40]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    colors = ["#42A5F5", "#FF7043"]

    # 参数量
    axes[0].bar(models, params, color=colors)
    axes[0].set_ylabel("Parameters (M)")
    axes[0].set_title("Model Size")
    for i, v in enumerate(params):
        axes[0].text(i, v + 0.3, f"{v}M", ha="center", fontsize=11)

    # GPU 显存
    axes[1].bar(models, gpu_mem, color=colors)
    axes[1].set_ylabel("GPU Memory (GB)")
    axes[1].set_title("GPU Memory Usage")
    for i, v in enumerate(gpu_mem):
        axes[1].text(i, v + 0.1, f"{v}GB", ha="center", fontsize=11)

    # 训练时间
    axes[2].bar(models, train_time, color=colors)
    axes[2].set_ylabel("Time (min)")
    axes[2].set_title("Training Time")
    for i, v in enumerate(train_time):
        axes[2].text(i, v + 0.5, f"{v}min", ha="center", fontsize=11)

    plt.suptitle("Resource Consumption Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig("resource_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: resource_comparison.png")


def plot_ema_effect():
    """展示 EMA 平滑效果的示意图（基于 SE-ResNet34 日志数据）。"""
    # SE-ResNet34 关键 epoch 数据
    epochs = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 79]
    raw_acc = [52.32, 66.40, 80.64, 79.96, 90.76, 88.00, 92.56, 90.48, 94.16, 95.92, 90.76, 95.32, 95.96, 97.12, 97.32, 98.04, 97.92]
    ema_acc = [50.44, 57.44, 66.72, 79.84, 93.48, 95.96, 96.72, 97.16, 97.52, 97.64, 97.64, 97.72, 98.28, 98.40, 98.44, 98.48, 98.56]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, raw_acc, "r-o", label="Raw Model", linewidth=1.5, markersize=4, alpha=0.8)
    ax.plot(epochs, ema_acc, "g-s", label="EMA Model", linewidth=2, markersize=5)

    # 标注关键点
    ax.annotate(f"Best: {max(ema_acc):.2f}%", xy=(79, 98.56),
                xytext=(65, 95), fontsize=11, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="green"),
                color="green")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=12)
    ax.set_title("EMA Smoothing Effect (SE-ResNet34)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("ema_effect.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: ema_effect.png")


if __name__ == "__main__":
    plot_model_comparison()
    plot_resource_comparison()
    plot_ema_effect()
