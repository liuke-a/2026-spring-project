"""
日志工具模块。

配置标准日志格式，同时输出到控制台和文件。
每次调用 setup_logger 都会生成带时间戳的新日志文件，避免覆盖历史记录。
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# 内置默认值，不再依赖 config_scratch.py
_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "cat_dog",
    level: int = logging.INFO,
    log_file: Path | str | None = None,
) -> logging.Logger:
    """
    初始化并返回一个配置好的 Logger。

    同时向控制台和日志文件输出，避免重复添加 Handler。
    每次调用默认会创建带时间戳的新日志文件。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 防止重复添加 Handler（多次 import 时）
    if logger.handlers:
        return logger

    # 日志文件路径：未指定则按时间戳生成
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = _LOG_DIR / f"train_{timestamp}.log"
    else:
        log_file = Path(log_file)

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # 文件输出
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(str(log_file), mode="w", encoding="utf-8")
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 在控制台打印日志文件路径，方便定位
    logger.info("Log file: %s", log_file)

    return logger