"""
从 train.log 提取训练曲线并绘图。
生成：
- training_curves_18.png：SE-ResNet18 训练曲线
- training_curves_34.png：SE-ResNet34 训练曲线
- training_curves_combined.png：两者对比
"""

import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

LOG_PATH = "../logs/train.log"

def parse_log(log_path):
    """解析 train.log，按训练轮次分组。"""
    runs = []
    current_run = None

    with open(log_path, "r") as f:
        for line in f:
            # 新训练轮次
            if "Scratch training (SE-ResNet) started" in line:
                current_run = {"config": "", "epochs": []}
                runs.append(current_run)
                continue

            if current_run is None:
                continue

            # 配置行
            m_cfg = re.search(r"Layers=\(([^)]+)\), Batch=(\d+), Epochs=(\d+), Warmup=(\d+), BaseLR=([\d.]+)", line)
            if m_cfg:
                current_run["config"] = {
                    "layers": m_cfg.group(1),
                    "batch": int(m_cfg.group(2)),
                    "epochs": int(m_cfg.group(3)),
                    "warmup": int(m_cfg.group(4)),
                    "base_lr": float(m_cfg.group(5)),
                }

            # Epoch 行
            m_epoch = re.search(r"Epoch \[(\d+)/(\d+)\]", line)
            if m_epoch:
                epoch_num = int(m_epoch.group(1))
                current_run["epochs"].append({"epoch": epoch_num})

            # Train 结果
            m_train = re.search(r"Train -> Loss: ([\d.]+), Acc\(approx\): ([\d.]+)%", line)
            if m_train and current_run["epochs"]:
                current_run["epochs"][-1]["train_loss"] = float(m_train.group(1))
                current_run["epochs"][-1]["train_acc"] = float(m_train.group(2))

            # Val(raw) 结果
            m_val_raw = re.search(r"Val\(raw\) -> Loss: ([\d.]+), Acc: ([\d.]+)%", line)
            if m_val_raw and current_run["epochs"]:
                current_run["epochs"][-1]["raw_loss"] = float(m_val_raw.group(1))
                current_run["epochs"][-1]["raw_acc"] = float(m_val_raw.group(2))

            # Val(EMA) 结果
            m_val_ema = re.search(r"Val\(EMA\) -> Loss: ([\d.]+), Acc: ([\d.]+)%", line)
            if m_val_ema and current_run["epochs"]:
                current_run["epochs"][-1]["ema_loss"] = float(m_val_ema.group(1))
                current_run["epochs"][-1]["ema_acc"] = float(m_val_ema.group(2))

    # 只保留有完整数据的 epoch
    for run in runs:
        run["epochs"] = [e for e in run["epochs"] if "train_loss" in e and "ema_acc" in e]

    return runs


def plot_single_run(run, title, save_path):
    """绘制单次训练的曲线。"""
    epochs = [e["epoch"] for e in run["epochs"]]
    train_loss = [e["train_loss"] for e in run["epochs"]]
    raw_loss = [e.get("raw_loss", None) for e in run["epochs"]]
    ema_loss = [e.get("ema_loss", None) for e in run["epochs"]]
    train_acc = [e["train_acc"] for e in run["epochs"]]
    raw_acc = [e.get("raw_acc", None) for e in run["epochs"]]
    ema_acc = [e.get("ema_acc", None) for e in run["epochs"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, train_loss, "b-", label="Train Loss", linewidth=1.5)
    ax1.plot(epochs, raw_loss, "r--", label="Val Raw Loss", linewidth=1.5)
    ax1.plot(epochs, ema_loss, "g--", label="Val EMA Loss", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, train_acc, "b-", label="Train Acc", linewidth=1.5)
    ax2.plot(epochs, raw_acc, "r--", label="Val Raw Acc", linewidth=1.5)
    ax2.plot(epochs, ema_acc, "g--", label="Val EMA Acc", linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy Curves")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    cfg = run.get("config", {})
    layers = cfg.get("layers", "?")
    fig.suptitle(f"{title} (layers=({layers}))", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_combined(runs, save_path):
    """绘制两个模型的 EMA Accuracy 对比。"""
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = ["SE-ResNet18", "SE-ResNet34"]
    colors = ["#2196F3", "#FF5722"]

    for i, run in enumerate(runs):
        epochs = [e["epoch"] for e in run["epochs"]]
        ema_acc = [e.get("ema_acc", None) for e in run["epochs"]]
        cfg = run.get("config", {})
        layers = cfg.get("layers", "?")
        ax.plot(epochs, ema_acc, color=colors[i], linewidth=2,
                label=f"{labels[i]} (layers=({layers}))")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("EMA Validation Accuracy (%)", fontsize=12)
    ax.set_title("SE-ResNet18 vs SE-ResNet34 Validation Accuracy", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    runs = parse_log(LOG_PATH)

    # 筛选成功的训练（Batch=256, BaseLR=0.2, >20 epochs）
    successful = [r for r in runs
                  if r.get("config") and r["config"].get("batch") == 256
                  and len(r["epochs"]) > 20]

    print(f"Found {len(successful)} runs with Batch=256 and >20 epochs")

    # 取每个架构的最后一次（最成功的）运行
    se18_run = None
    se34_run = None
    layers_18 = "2, 2, 2, 2"
    layers_34 = "3, 4, 6, 3"
    for r in successful:
        if r["config"]["layers"] == layers_18:
            se18_run = r
        elif r["config"]["layers"] == layers_34:
            se34_run = r

    if se18_run:
        plot_single_run(se18_run, "SE-ResNet18", "training_curves_18.png")
    if se34_run:
        plot_single_run(se34_run, "SE-ResNet34", "training_curves_34.png")
    if se18_run and se34_run:
        plot_combined([se18_run, se34_run], "training_curves_combined.png")
