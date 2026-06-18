"""
模型服务层 —— 猫狗图像分类对外推理接口。

提供统一的模型加载、预处理和推理抽象，支持扩展不同的模型架构。

Quick start:
    # 方式一：通过工厂获取（推荐，从 service/config.yaml 读取配置）
    from service import get_model_service

    svc = get_model_service()
    result = svc.predict("data/images/cat.0.jpg")
    print(result["label"], result["confidence"])
    # → cat 0.9748

    # 方式二：直接实例化特定模型
    from service import SEResNet18Service

    svc = SEResNet18Service()
    results = svc.predict_batch(["data/test/cat.1.jpg", "data/test/dog.1.jpg"])
    for r in results:
        print(r["label"], r["confidence"])
"""

from service.base import BaseModelService, CAT_LABEL, DOG_LABEL, LABEL_MAP
from service.seresnet18 import SEResNet18Service
from service.seresnet34 import SEResNet34Service
from service.seresnet34_big import SEResNet34BigService
from service.cnn import CNNService
from service.factory import get_model_service, get_available_models

__all__ = [
    "BaseModelService",
    "SEResNet18Service",
    "SEResNet34Service",
    "SEResNet34BigService",
    "CNNService",
    "get_model_service",
    "get_available_models",
    "CAT_LABEL",
    "DOG_LABEL",
    "LABEL_MAP",
]
