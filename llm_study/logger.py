"""日志模块：提供统一的日志记录功能，支持从 .env 文件加载配置。"""
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def load_env_file(env_path: Optional[Path] = None) -> dict:
    """
    从 .env 文件加载环境变量（简单实现，不依赖 python-dotenv）。
    
    Args:
        env_path: .env 文件路径，默认查找 llm_study/.env
    
    Returns:
        环境变量字典
    """
    if env_path is None:
        # 默认查找 llm_study/.env
        env_path = Path(__file__).parent / ".env"
    
    env_vars = {}
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # 跳过空行和注释
                if not line or line.startswith("#"):
                    continue
                # 解析 KEY=VALUE
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    env_vars[key] = value
    
    return env_vars


def setup_logger(
    name: str = "llm_study",
    level: Optional[str] = None,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
    env_path: Optional[Path] = None,
) -> logging.Logger:
    """
    设置并返回 logger 实例。
    
    Args:
        name: logger 名称，默认 "llm_study"
        level: 日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL），默认从 .env 读取或 INFO
        log_file: 日志文件路径，如果提供则同时输出到文件和控制台
        format_string: 日志格式字符串，默认包含时间、级别、名称、消息
        env_path: .env 文件路径
    
    Returns:
        配置好的 logger 实例
    """
    # 加载环境变量
    env_vars = load_env_file(env_path)
    
    # 确定日志级别
    if level is None:
        level = env_vars.get("LOG_LEVEL", "INFO").upper()
    
    log_level = getattr(logging, level, logging.INFO)
    
    # 确定日志文件路径
    if log_file is None and "LOG_FILE" in env_vars:
        log_file = Path(env_vars["LOG_FILE"])
    
    # 确定日志格式
    if format_string is None:
        format_string = env_vars.get(
            "LOG_FORMAT",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    # 创建 logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # 避免重复添加 handler
    if logger.handlers:
        return logger
    
    # 创建 formatter
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    
    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件 handler（如果指定）
    if log_file:
        # 在文件名中添加时间戳
        log_file = Path(log_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 分离文件名和扩展名，插入时间戳
        stem = log_file.stem  # 文件名（不含扩展名）
        suffix = log_file.suffix  # 扩展名（如 .log）
        log_file_with_timestamp = log_file.parent / f"{stem}_{timestamp}{suffix}"
        
        log_file_with_timestamp.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file_with_timestamp, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# 默认 logger 实例
_default_logger: Optional[logging.Logger] = None


def get_logger(name: str = "llm_study", **kwargs) -> logging.Logger:
    """
    获取 logger 实例（单例模式）。
    
    Args:
        name: logger 名称
        **kwargs: 传递给 setup_logger 的其他参数
    
    Returns:
        logger 实例
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logger(name, **kwargs)
    return _default_logger


# 便捷函数
def debug(msg: str, *args, **kwargs):
    """记录 DEBUG 级别日志。"""
    get_logger().debug(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs):
    """记录 INFO 级别日志。"""
    get_logger().info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs):
    """记录 WARNING 级别日志。"""
    get_logger().warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs):
    """记录 ERROR 级别日志。"""
    get_logger().error(msg, *args, **kwargs)


def critical(msg: str, *args, **kwargs):
    """记录 CRITICAL 级别日志。"""
    get_logger().critical(msg, *args, **kwargs)
