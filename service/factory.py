"""
模型服务工厂。

读取 service/config.yaml，根据配置返回对应的 BaseModelService 子类实例。

Usage:
    from service import get_model_service

    svc = get_model_service()                        # 使用 config.yaml 默认设置
    svc = get_model_service("seresnet34")            # 显式指定模型名
    result = svc.predict("data/images/cat.0.jpg")
"""

import logging
from pathlib import Path
from typing import Optional, Union

import yaml

from service.base import BaseModelService

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"

# 模型名到 Service 类的延迟导入映射
_SERVICE_CLASSES = {
    "seresnet18": ("service.seresnet18", "SEResNet18Service"),
    "seresnet34": ("service.seresnet34", "SEResNet34Service"),
}


def _load_config(config_path: Optional[Union[str, Path]] = None) -> dict:
    """加载并返回 YAML 配置。"""
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Service config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_model_service(
    model: Optional[str] = None,
    config_path: Optional[Union[str, Path]] = None,
) -> BaseModelService:
    """
    根据配置获取分类模型服务实例。

    优先级：传入 model > YAML 配置 service.model。

    Args:
        model: 模型名，可选 "seresnet18" | "seresnet34"。None 则从 YAML 读取。
        config_path: YAML 配置文件路径，None 则使用 service/config.yaml。

    Returns:
        BaseModelService 子类实例（已加载权重，处于 eval 模式）。

    Raises:
        FileNotFoundError: 配置文件或权重文件不存在。
        ValueError: 模型名未注册。
    """
    cfg = _load_config(config_path)

    model_name = model or cfg["service"]["model"]
    device = cfg["service"].get("device", "auto")
    device = None if device == "auto" else device

    if model_name not in cfg.get("models", {}):
        raise ValueError(
            f"模型 '{model_name}' 未在配置文件中定义。"
            f"可用模型: {list(cfg.get('models', {}).keys())}"
        )

    model_cfg = cfg["models"][model_name]
    weights = model_cfg["weights"]
    # 权重路径支持相对于项目根目录
    weights_path = Path(weights)
    if not weights_path.is_absolute():
        weights_path = PROJECT_ROOT / weights_path

    if not weights_path.exists():
        raise FileNotFoundError(
            f"权重文件不存在: {weights_path}\n"
            f"请确认模型 '{model_name}' 已训练并导出权重。"
        )

    # 延迟导入 Service 类
    if model_name not in _SERVICE_CLASSES:
        raise ValueError(
            f"模型 '{model_name}' 未注册 Service 类。"
            f"已注册: {list(_SERVICE_CLASSES.keys())}"
        )

    module_path, class_name = _SERVICE_CLASSES[model_name]
    import importlib
    module = importlib.import_module(module_path)
    service_cls = getattr(module, class_name)

    # 读取批量推理配置
    max_batch_size = cfg["service"].get("max_batch_size", 128)

    svc = service_cls(weights_path=weights_path, device=device)
    svc.max_batch_size = max_batch_size

    logger.info(
        f"Factory: model={model_name}, weights={weights_path}, "
        f"device={device or 'auto'}, max_batch_size={max_batch_size}"
    )
    return svc
