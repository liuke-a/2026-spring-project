"""
训练脚本。

支持：
- 混合精度训练 (AMP)
- 分层学习率 (Backbone 1e-4, Head 1e-3)
- 冻结 Warmup (前 3 epoch 只训练 Head)
- 余弦退火学习率调度
- 早停 (Early Stopping)
- TensorBoard 监控
- 断点续训 (保存/恢复 optimizer + scheduler + epoch)
"""

import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

# 将项目根目录加入路径，确保能 import 本地模块
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    TRAIN_DIR, CHECKPOINT_DIR, LOG_DIR, TENSORBOARD_DIR,
    IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, VAL_RATIO, RANDOM_SEED,
    NUM_EPOCHS, WARMUP_EPOCHS, EARLY_STOP_PATIENCE,
    LR_BACKBONE, LR_HEAD, WEIGHT_DECAY, DEVICE
)
from utils.logger import setup_logger
from utils.dataset import get_dataloaders
from models.classifier import CatDogClassifier


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_acc: float,
    path: Path,
    is_best: bool = False
) -> None:
    """保存训练断点（含模型、优化器、调度器状态）。"""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': best_acc,
    }
    torch.save(state, path)
    if is_best:
        best_path = path.parent / 'best_model.pth'
        torch.save(state, best_path)
        logging.getLogger(__name__).info(f"Best model saved -> {best_path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    path: Path
) -> int:
    """从断点恢复训练状态，返回起始 epoch。"""
    logger = logging.getLogger(__name__)
    if not path.exists():
        logger.warning(f"No checkpoint found at {path}, starting from scratch.")
        return 0

    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_acc = checkpoint['best_acc']

    logger.info(f"Resumed from epoch {start_epoch - 1}, best_acc={best_acc:.4f}")
    return start_epoch


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    epoch: int
) -> tuple[float, float]:
    """训练一个 epoch，返回平均 loss 和准确率。"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # 混合精度前向
        with autocast():
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

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion,
    device
) -> tuple[float, float]:
    """验证一个 epoch，返回平均 loss 和准确率。"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def main():
    # 初始化日志
    logger = setup_logger("train")
    logger.info("=" * 50)
    logger.info("Training started")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Epochs: {NUM_EPOCHS}, Warmup: {WARMUP_EPOCHS}, Batch: {BATCH_SIZE}")

    # 固定随机种子
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    # 数据加载
    logger.info("Loading datasets...")
    train_loader, val_loader = get_dataloaders()

    # 模型构建
    logger.info("Building model...")
    model = CatDogClassifier(num_classes=2, pretrained=True).to(DEVICE)

    # 初始冻结 Backbone（Warmup 阶段）
    for param in model.backbone.parameters():
        param.requires_grad = False
    logger.info(f"Backbone frozen for first {WARMUP_EPOCHS} epochs")

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 优化器：分层学习率，包含所有参数组（冻结部分不计算梯度）
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': LR_BACKBONE},
        {'params': model.head.parameters(), 'lr': LR_HEAD}
    ], weight_decay=WEIGHT_DECAY)

    # 学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    # AMP 混合精度
    scaler = GradScaler()

    # TensorBoard
    writer = SummaryWriter(log_dir=TENSORBOARD_DIR)

    # 断点续训
    resume_path = CHECKPOINT_DIR / 'resume.pth'
    start_epoch = 0
    best_acc = 0.0
    patience_counter = 0

    # 如果命令行带 --resume 参数，尝试恢复
    if '--resume' in sys.argv and resume_path.exists():
        start_epoch = load_checkpoint(model, optimizer, scheduler, resume_path)
        # 恢复后需要重新判断 warmup 状态
        if start_epoch >= WARMUP_EPOCHS:
            for param in model.backbone.parameters():
                param.requires_grad = True
            logger.info("Backbone already unfrozen (resumed after warmup)")

    # 训练循环
    for epoch in range(start_epoch, NUM_EPOCHS):
        logger.info(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")

        # Warmup 解冻判断
        if epoch == WARMUP_EPOCHS:
            logger.info("Unfreezing backbone for fine-tuning...")
            for param in model.backbone.parameters():
                param.requires_grad = True

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, DEVICE, epoch
        )
        logger.info(f"  Train -> Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")

        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        logger.info(f"  Val   -> Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        # 学习率步进
        scheduler.step()

        # TensorBoard 记录
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Checkpoint 保存（每轮都存 resume，方便断点续训）
        save_checkpoint(
            model, optimizer, scheduler, epoch, best_acc,
            resume_path, is_best=False
        )

        # 最佳模型判断
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_acc,
                CHECKPOINT_DIR / 'best_model.pth', is_best=True
            )
            logger.info(f"  *** New best accuracy: {best_acc:.2f}% ***")
        else:
            patience_counter += 1
            logger.info(f"  EarlyStop patience: {patience_counter}/{EARLY_STOP_PATIENCE}")

        # 早停
        if patience_counter >= EARLY_STOP_PATIENCE:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    writer.close()
    logger.info(f"Training finished. Best validation accuracy: {best_acc:.2f}%")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()