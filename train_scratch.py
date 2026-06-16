"""
从头训练脚本（SE-ResNet）— RTX 3080Ti 12GB 优化版。

核心配方：
- 单一参数组 SGD(momentum, nesterov) + 线性 warmup + 余弦退火
- 混合精度 (AMP) + channels_last + TF32 + torch.compile（3080Ti 全栈加速）
- Mixup / CutMix（按 batch 随机切换）+ Label Smoothing 的 soft-target 损失
- 模型权重 EMA（用 EMA 权重做验证并保存 best）
- 早停 + TensorBoard + 断点续训 (--resume)
- persistent_workers + prefetch_factor（消除数据供给瓶颈）

输出写入 checkpoints/scratch/，不会覆盖预训练流程的 checkpoints。
"""

import argparse
import copy
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config_scratch import (
    TRAIN_DIR, SCRATCH_CHECKPOINT_DIR, SCRATCH_TENSORBOARD_DIR,
    RANDOM_SEED, NUM_WORKERS, VAL_RATIO, MAX_SAMPLES,
    MODEL_LAYERS, MODEL_WIDTH, DROP_RATE, DROP_PATH,
    BATCH_SIZE, NUM_EPOCHS, WARMUP_EPOCHS, EARLY_STOP_PATIENCE,
    BASE_LR, MIN_LR, MOMENTUM, NESTEROV, WEIGHT_DECAY,
    LABEL_SMOOTHING, RANDOM_ERASING_PROB, RAND_AUGMENT_N, RAND_AUGMENT_M,
    MIXUP_ALPHA, CUTMIX_ALPHA, MIXUP_PROB, MIXUP_SWITCH_PROB,
    EMA_DECAY, DEVICE,
)
from utils.logger import setup_logger
from utils.dataset import get_dataloaders
from models.seresnet import build_scratch_model


def _gpu_memory() -> str:
    """返回 GPU 显存使用情况的简短字符串。"""
    if not torch.cuda.is_available():
        return ""
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    return f"GPU={allocated:.1f}/{reserved:.1f}G"


def _unwrap_model(model: nn.Module) -> nn.Module:
    """解包 torch.compile 包裹，便于访问真实模块。"""
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def _find_non_finite_buffers(model: nn.Module) -> list[str]:
    """返回含有 nan/inf 的浮点 buffer 名称列表。"""
    bad_buffers = []
    for name, buf in _unwrap_model(model).named_buffers():
        if torch.is_floating_point(buf) and not torch.isfinite(buf).all():
            bad_buffers.append(name)
    return bad_buffers


class ModelEMA:
    """模型权重的指数滑动平均（EMA），通常带来更平滑、更高的验证精度。"""

    def __init__(self, model: nn.Module, decay: float = 0.9998) -> None:
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        # 对所有浮点 state（参数 + BN running stats）统一做 EMA，
        # 避免参数是旧值而 BN buffer 被直接拷贝成新值导致 eval 失配。
        src = _unwrap_model(model)
        d = self.decay
        ema_state = self.ema.state_dict()
        src_state = src.state_dict()
        for k, v in src_state.items():
            if torch.is_floating_point(v):
                ema_state[k].mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                ema_state[k].copy_(v)


def rand_bbox(width: int, height: int, lam: float):
    """为 CutMix 生成随机裁剪框。"""
    cut_rat = math.sqrt(1.0 - lam)
    cut_w, cut_h = int(width * cut_rat), int(height * cut_rat)
    cx, cy = np.random.randint(width), np.random.randint(height)
    x1 = np.clip(cx - cut_w // 2, 0, width)
    y1 = np.clip(cy - cut_h // 2, 0, height)
    x2 = np.clip(cx + cut_w // 2, 0, width)
    y2 = np.clip(cy + cut_h // 2, 0, height)
    return x1, y1, x2, y2


def apply_mixup_cutmix(images: torch.Tensor, labels: torch.Tensor):
    """
    按概率对一个 batch 施加 Mixup 或 CutMix。

    Returns:
        (images, labels_a, labels_b, lam)  其中损失 = lam*L(a) + (1-lam)*L(b)
    """
    if np.random.rand() > MIXUP_PROB:
        return images, labels, labels, 1.0

    perm = torch.randperm(images.size(0), device=images.device)
    labels_b = labels[perm]

    use_cutmix = np.random.rand() < MIXUP_SWITCH_PROB
    if use_cutmix:
        lam = np.random.beta(CUTMIX_ALPHA, CUTMIX_ALPHA)
        _, _, h, w = images.shape
        x1, y1, x2, y2 = rand_bbox(w, h, lam)
        images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
        # 按实际裁剪面积修正 lam
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (w * h))
    else:
        lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
        images = lam * images + (1.0 - lam) * images[perm]

    return images, labels, labels_b, float(lam)


def soft_ce_loss(outputs, labels_a, labels_b, lam, label_smoothing: float):
    """Mixup/CutMix 的混合交叉熵（带 label smoothing）。"""
    loss_a = F.cross_entropy(outputs, labels_a, label_smoothing=label_smoothing)
    loss_b = F.cross_entropy(outputs, labels_b, label_smoothing=label_smoothing)
    return lam * loss_a + (1.0 - lam) * loss_b


def build_scheduler(optimizer, steps_per_epoch: int):
    """线性 warmup + 余弦退火（按 step 粒度）。"""
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    total_steps = NUM_EPOCHS * steps_per_epoch
    min_ratio = MIN_LR / BASE_LR

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, ema, optimizer, scheduler, scaler, epoch, best_acc, path, is_best=False):
    """保存断点（含模型、EMA、优化器、调度器、scaler 状态）。"""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema.ema.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_acc': best_acc,
    }
    torch.save(state, path)
    if is_best:
        best_path = path.parent / 'best_model.pth'
        torch.save(state, best_path)
        logging.getLogger(__name__).info(f"Best model saved -> {best_path}")


def load_checkpoint(model, ema, optimizer, scheduler, scaler, path):
    """从断点恢复，返回 (start_epoch, best_acc)。"""
    logger = logging.getLogger(__name__)
    if not path.exists():
        logger.warning(f"No checkpoint found at {path}, starting from scratch.")
        return 0, 0.0

    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    ema.ema.load_state_dict(ckpt['ema_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    scaler.load_state_dict(ckpt['scaler_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    best_acc = ckpt['best_acc']
    logger.info(f"Resumed from epoch {start_epoch - 1}, best_acc={best_acc:.4f}")
    return start_epoch, best_acc


def train_one_epoch(model, ema, loader, optimizer, scheduler, scaler, device,
                    epoch: int = 0, num_epochs: int = 1):
    """训练一个 epoch，返回平均 loss 和（近似）训练准确率。"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Train [{epoch+1}/{num_epochs}]", unit="batch",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}] "
                           "{postfix}")
    for images, labels in pbar:
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)

        images, labels_a, labels_b, lam = apply_mixup_cutmix(images, labels)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            outputs = model(images)
            loss = soft_ce_loss(outputs, labels_a, labels_b, lam, LABEL_SMOOTHING)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(model)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        # 以主标签近似统计训练准确率（mixup 下仅供参考）
        correct += predicted.eq(labels_a).sum().item()
        total += labels_a.size(0)

        # 实时更新进度条：当前 loss + 显存
        avg_loss = total_loss / (pbar.n or 1)
        gpu_str = _gpu_memory()
        lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix_str(f"loss={avg_loss:.4f} lr={lr:.2e} {gpu_str}")

    return total_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def validate(model, loader, device, epoch: int = 0, num_epochs: int = 1, desc: str = "Val"):
    """用给定模型在验证集评估，返回 loss、准确率和诊断信息。"""
    logger = logging.getLogger(__name__)
    model = _unwrap_model(model)
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    batch_count = 0
    non_finite_batches = 0
    pred_counts = torch.zeros(2, dtype=torch.long)

    pbar = tqdm(loader, desc=f"{desc} [{epoch+1}/{num_epochs}]", unit="batch",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}] "
                           "{postfix}")
    for batch_idx, (images, labels) in enumerate(pbar, start=1):
        batch_count += 1
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)

        with autocast():
            outputs = model(images)
        outputs_fp32 = outputs.float()
        loss = F.cross_entropy(outputs_fp32, labels)

        outputs_finite = bool(torch.isfinite(outputs_fp32).all().item())
        loss_finite = bool(torch.isfinite(loss).item())
        if not outputs_finite or not loss_finite:
            non_finite_batches += 1
            if non_finite_batches <= 3:
                logger.warning(
                    f"{desc} batch {batch_idx}: detected non-finite values "
                    f"(outputs_finite={outputs_finite}, loss_finite={loss_finite})"
                )

        safe_outputs = torch.nan_to_num(outputs_fp32, nan=0.0, posinf=1e4, neginf=-1e4)
        if loss_finite:
            total_loss += loss.item()
        _, predicted = safe_outputs.max(1)
        pred_counts += torch.bincount(predicted, minlength=2).cpu()
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        avg_loss = float("nan") if non_finite_batches else total_loss / max(batch_count, 1)
        gpu_str = _gpu_memory()
        pbar.set_postfix_str(f"loss={avg_loss:.4f} {gpu_str}")

    bad_buffers = _find_non_finite_buffers(model)
    if bad_buffers:
        logger.warning(f"{desc}: found non-finite buffers: {', '.join(bad_buffers[:8])}")

    final_loss = float("nan") if non_finite_batches else total_loss / max(batch_count, 1)
    stats = {
        "pred_counts": pred_counts.tolist(),
        "non_finite_batches": non_finite_batches,
        "bad_buffers": bad_buffers,
    }
    return final_loss, 100.0 * correct / total, stats


def main():
    parser = argparse.ArgumentParser(description="Train SE-ResNet from scratch.")
    parser.add_argument('--resume', action='store_true', help='从 resume.pth 恢复训练')
    args = parser.parse_args()

    logger = setup_logger("train_scratch")
    logger.info("=" * 50)
    logger.info("Scratch training (SE-ResNet) started")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Layers={MODEL_LAYERS}, Batch={BATCH_SIZE}, Epochs={NUM_EPOCHS}, "
                f"Warmup={WARMUP_EPOCHS}, BaseLR={BASE_LR}")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.benchmark = True
        # TF32：3080Ti (Ada Lovelace) tensor core 加速，matmul ~2x，精度损失可忽略
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    logger.info("Loading datasets (strong augmentation, persistent workers)...")
    train_loader, val_loader = get_dataloaders(
        max_samples=MAX_SAMPLES,
        strong=True,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        rand_augment=(RAND_AUGMENT_N, RAND_AUGMENT_M),
        erasing_prob=RANDOM_ERASING_PROB,
        persistent_workers=True,
        prefetch_factor=2,
    )

    logger.info("Building SE-ResNet...")
    model = build_scratch_model(
        num_classes=2, layers=MODEL_LAYERS, width=MODEL_WIDTH,
        drop_rate=DROP_RATE, drop_path=DROP_PATH,
    ).to(DEVICE).to(memory_format=torch.channels_last)

    # EMA 基于未编译的模型创建，避免验证时 CUDA graph 动态 batch size 问题
    ema = ModelEMA(model, decay=EMA_DECAY)

    # torch.compile：(SM 8.9) 支持，kernel fusion 提速 10-30%
    # default 模式兼容 autocast + GradScaler + mixup；仅训练用，EMA 保持 eager
    try:
        model = torch.compile(model, mode="default")
        logger.info("torch.compile enabled (mode=default)")
    except Exception as e:
        logger.warning(f"torch.compile failed, falling back to eager: {e}")

    optimizer = optim.SGD(
        model.parameters(), lr=BASE_LR, momentum=MOMENTUM,
        nesterov=NESTEROV, weight_decay=WEIGHT_DECAY,
    )
    scheduler = build_scheduler(optimizer, steps_per_epoch=len(train_loader))
    scaler = GradScaler()
    writer = SummaryWriter(log_dir=str(SCRATCH_TENSORBOARD_DIR))

    resume_path = SCRATCH_CHECKPOINT_DIR / 'resume.pth'
    start_epoch, best_acc = 0, 0.0
    patience_counter = 0

    if args.resume:
        start_epoch, best_acc = load_checkpoint(
            model, ema, optimizer, scheduler, scaler, resume_path
        )

    # 打印 GPU 显存基线
    logger.info(f"GPU memory before training: {_gpu_memory()}")

    for epoch in range(start_epoch, NUM_EPOCHS):
        logger.info(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")

        train_loss, train_acc = train_one_epoch(
            model, ema, train_loader, optimizer, scheduler, scaler, DEVICE,
            epoch=epoch, num_epochs=NUM_EPOCHS,
        )
        logger.info(f"  Train -> Loss: {train_loss:.4f}, Acc(approx): {train_acc:.2f}%")

        raw_model = _unwrap_model(model)
        raw_val_loss, raw_val_acc, raw_stats = validate(
            raw_model, val_loader, DEVICE,
            epoch=epoch, num_epochs=NUM_EPOCHS, desc="Val(raw)"
        )
        logger.info(f"  Val(raw) -> Loss: {raw_val_loss:.4f}, Acc: {raw_val_acc:.2f}%")
        logger.info(
            f"  Val(raw) PredDist -> cat={raw_stats['pred_counts'][0]}, "
            f"dog={raw_stats['pred_counts'][1]}"
        )

        ema_val_loss, ema_val_acc, ema_stats = validate(
            ema.ema, val_loader, DEVICE,
            epoch=epoch, num_epochs=NUM_EPOCHS, desc="Val(EMA)"
        )
        logger.info(f"  Val(EMA) -> Loss: {ema_val_loss:.4f}, Acc: {ema_val_acc:.2f}%")
        logger.info(
            f"  Val(EMA) PredDist -> cat={ema_stats['pred_counts'][0]}, "
            f"dog={ema_stats['pred_counts'][1]}"
        )

        if raw_stats["non_finite_batches"]:
            logger.warning(
                f"  Val(raw) NonFinite -> batches={raw_stats['non_finite_batches']}"
            )
        if ema_stats["non_finite_batches"]:
            logger.warning(
                f"  Val(EMA) NonFinite -> batches={ema_stats['non_finite_batches']}"
            )

        monitor_loss, monitor_acc = ema_val_loss, ema_val_acc
        if not math.isfinite(ema_val_loss):
            monitor_loss, monitor_acc = raw_val_loss, raw_val_acc
            logger.warning(
                "EMA validation is non-finite; fallback to raw validation for "
                "early stopping and best-checkpoint selection."
            )

        cur_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', monitor_loss, epoch)
        writer.add_scalar('Loss/val_raw', raw_val_loss, epoch)
        writer.add_scalar('Loss/val_ema', ema_val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', monitor_acc, epoch)
        writer.add_scalar('Accuracy/val_raw', raw_val_acc, epoch)
        writer.add_scalar('Accuracy/val_ema', ema_val_acc, epoch)
        writer.add_scalar('LR', cur_lr, epoch)

        save_checkpoint(
            model, ema, optimizer, scheduler, scaler, epoch, best_acc, resume_path
        )

        if monitor_acc > best_acc:
            best_acc = monitor_acc
            patience_counter = 0
            save_checkpoint(
                model, ema, optimizer, scheduler, scaler, epoch, best_acc,
                resume_path, is_best=True
            )
            logger.info(f"  *** New best accuracy: {best_acc:.2f}% ***")
        else:
            patience_counter += 1
            logger.info(f"  EarlyStop patience: {patience_counter}/{EARLY_STOP_PATIENCE}")

        if patience_counter >= EARLY_STOP_PATIENCE:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    writer.close()
    logger.info(f"Scratch training finished. Best validation accuracy: {best_acc:.2f}%")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
