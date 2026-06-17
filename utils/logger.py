"""
日志工具模块。

配置标准日志格式，同时输出到控制台和文件。
"""

import logging
import sys
from pathlib import Path

from config_scratch import LOG_FORMAT, LOG_DATE_FORMAT, LOG_DIR, LOG_FILE


def setup_logger(name: str = "cat_dog", level: int = logging.INFO) -> logging.Logger:
    """
    初始化并返回一个配置好的 Logger。

    同时向控制台和日志文件输出，避免重复添加 Handler。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 防止重复添加 Handler（多次 import 时）
    if logger.handlers:
        return logger

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # 文件输出
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger