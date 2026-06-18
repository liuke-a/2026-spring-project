"""
SimpleCNN 训练脚本。

支持：
- 混合精度训练 (AMP)
- AdamW 优化器 + 余弦退火学习率调度
- tqdm 进度条（每 epoch 实时展示 batch 级进度）
- 早停 (Early Stopping)
- TensorBoard 监控
- 断点续训 (保存/恢复 optimizer + scheduler + epoch)
- 训练曲线图（Loss / Accuracy / LR，自动保存为 PNG）

Usage:
    python models/CNN/train.py              # 从头训练
    python models/CNN/train.py --resume     # 从断点恢复
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 非交互后端，确保 headless 环境正常保存
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 将项目根目录加入路径，确保能 import 本地模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.CNN.config import (
    CNN_CHECKPOINT_DIR, CNN_TENSORBOARD_DIR,
    BATCH_SIZE, NUM_WORKERS, RANDOM_SEED,
    NUM_EPOCHS, EARLY_STOP_PATIENCE,
    CNN_LR, CNN_DROPOUT, WEIGHT_DECAY, DEVICE, MAX_SAMPLES,
)
from utils.logger import setup_logger
from utils.dataset import get_dataloaders
from models.CNN.CNN import build_cnn_model

# 训练曲线输出目录（CNN 模块内部）
_CNN_PLOT_DIR = Path(__file__).resolve().parent


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_acc: float,
    path: Path,
) -> None:
    """保存训练断点（含模型、优化器、调度器状态）。"""
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_acc": best_acc,
    }
    torch.save(state, path)


def save_best_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_acc: float,
    path: Path,
) -> None:
    """保存最佳模型断点。"""
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_acc": best_acc,
    }
    torch.save(state, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    path: Path,
) -> tuple[int, float]:
    """从断点恢复训练状态，返回 (start_epoch, best_acc)。"""
    logger = logging.getLogger(__name__)
    if not path.exists():
        logger.warning(f"No checkpoint found at {path}, starting from scratch.")
        return 0, 0.0

    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_acc = checkpoint["best_acc"]

    logger.info(f"Resumed from epoch {start_epoch - 1}, best_acc={best_acc:.4f}")
    return start_epoch, best_acc


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    """训练一个 epoch，返回平均 loss 和准确率。"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Train [{epoch}/{total_epochs}]", unit="batch")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # 混合精度前向
        with autocast(DEVICE):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # 混合精度反向
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 统计
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        # 实时更新进度条
        pbar.set_postfix({
            "loss": f"{loss.item():.3f}",
            "acc": f"{100.0 * correct / total:.1f}%",
        })

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion,
    device,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    """验证一个 epoch，返回平均 loss 和准确率。"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Val   [{epoch}/{total_epochs}]", unit="batch")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        with autocast(DEVICE):
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            "loss": f"{loss.item():.3f}",
            "acc": f"{100.0 * correct / total:.1f}%",
        })

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    train_accs: list,
    val_accs: list,
    lrs: list,
    save_dir: Path,
    best_epoch: int,
) -> None:
    """
    绘制并保存训练曲线：Loss、Accuracy、Learning Rate 三合一图。
    """
    epochs = list(range(1, len(train_losses) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Loss ---
    axes[0].plot(epochs, train_losses, "b-o", markersize=4, label="Train Loss")
    axes[0].plot(epochs, val_losses, "r-o", markersize=4, label="Val Loss")
    axes[0].axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.7,
                    label=f"Best epoch={best_epoch}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Accuracy ---
    axes[1].plot(epochs, train_accs, "b-o", markersize=4, label="Train Acc")
    axes[1].plot(epochs, val_accs, "r-o", markersize=4, label="Val Acc")
    axes[1].axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.7,
                    label=f"Best epoch={best_epoch}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy Curves")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # --- Learning Rate ---
    axes[2].plot(epochs, lrs, "g-o", markersize=4)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale("log")

    plt.tight_layout()
    save_path = save_dir / "training_curves_cnn.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 同时输出最佳 epoch 的摘要文本
    summary_path = save_dir / "training_summary_cnn.txt"
    lines = [
        "SimpleCNN Training Summary",
        "=" * 35,
        f"Total epochs:    {len(train_losses)}",
        f"Best epoch:      {best_epoch}",
        f"Best Val Acc:    {val_accs[best_epoch - 1]:.2f}%",
        f"Final Train Loss: {train_losses[-1]:.4f}",
        f"Final Val Loss:   {val_losses[-1]:.4f}",
        f"Final Train Acc:  {train_accs[-1]:.2f}%",
        f"Final Val Acc:    {val_accs[-1]:.2f}%",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    # 用 root logger 输出（该 logger 已由 setup_logger 配置好 handler）
    logging.getLogger("cnn_train").info(f"Curves saved to: {save_path}")
    logging.getLogger("cnn_train").info(f"Summary saved to: {summary_path}")


def main():
    # 初始化日志
    logger = setup_logger("cnn_train")
    logger.info("=" * 50)
    logger.info("SimpleCNN Training started")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Epochs: {NUM_EPOCHS}, Batch: {BATCH_SIZE}, LR: {CNN_LR}")

    # 固定随机种子
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    # 数据加载
    logger.info("Loading datasets...")
    train_loader, val_loader = get_dataloaders(
        max_samples=MAX_SAMPLES,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # 模型构建
    logger.info("Building model...")
    model = build_cnn_model(num_classes=2, dropout=CNN_DROPOUT).to(DEVICE)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 优化器：AdamW
    optimizer = optim.AdamW(
        model.parameters(), lr=CNN_LR, weight_decay=WEIGHT_DECAY
    )

    # 学习率调度：余弦退火
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    # AMP 混合精度
    scaler = GradScaler(DEVICE)

    # TensorBoard
    writer = SummaryWriter(log_dir=CNN_TENSORBOARD_DIR)

    # 断点续训
    resume_path = CNN_CHECKPOINT_DIR / "resume.pth"
    best_model_path = CNN_CHECKPOINT_DIR / "best_model.pth"
    start_epoch = 0
    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    # 训练历史记录
    train_losses: list = []
    val_losses: list = []
    train_accs: list = []
    val_accs: list = []
    lrs: list = []

    if "--resume" in sys.argv and resume_path.exists():
        start_epoch, best_acc = load_checkpoint(model, optimizer, scheduler, resume_path)

    # 训练循环
    for epoch in range(start_epoch, NUM_EPOCHS):
        current_epoch_label = epoch + 1
        logger.info(f"Epoch [{current_epoch_label}/{NUM_EPOCHS}]")

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, DEVICE,
            epoch=current_epoch_label, total_epochs=NUM_EPOCHS,
        )
        logger.info(f"  Train -> Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")

        # 验证
        val_loss, val_acc = validate(
            model, val_loader, criterion, DEVICE,
            epoch=current_epoch_label, total_epochs=NUM_EPOCHS,
        )
        logger.info(f"  Val   -> Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        # 学习率步进
        scheduler.step()

        # 记录当前 LR 与历史指标
        current_lr = optimizer.param_groups[0]["lr"]
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        lrs.append(current_lr)

        # TensorBoard 记录
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        # Checkpoint 保存（每轮都存 resume，方便断点续训）
        save_checkpoint(
            model, optimizer, scheduler, epoch, best_acc, resume_path
        )

        # 最佳模型判断
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = current_epoch_label
            patience_counter = 0
            save_best_checkpoint(
                model, optimizer, scheduler, epoch, best_acc, best_model_path
            )
            logger.info(f"  *** New best accuracy: {best_acc:.2f}% ***")
        else:
            patience_counter += 1
            logger.info(
                f"  EarlyStop patience: {patience_counter}/{EARLY_STOP_PATIENCE}"
            )

        # 早停
        if patience_counter >= EARLY_STOP_PATIENCE:
            logger.info(f"Early stopping triggered at epoch {current_epoch_label}")
            break

    writer.close()
    logger.info(f"Training finished. Best validation accuracy: {best_acc:.2f}%")

    # 绘制并保存训练曲线
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs, lrs,
        save_dir=_CNN_PLOT_DIR, best_epoch=best_epoch,
    )
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
