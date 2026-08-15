"""
记录程序运行过程中发生了什么，帮助你调试、定位错误、观察 Agent 的执行流程，并保留运行证据。
"""

import logging
import os
from utils.path_tool import get_abs_path
from datetime import datetime

# 日志保存的根目录
LOG_ROOT = get_abs_path("logs")

# 确保日志的目录存在
# 不存在创建
os.makedirs(LOG_ROOT, exist_ok=True)

# 日志的格式配置
# error info debug
DEFAULT_LOG_FORMAT = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)

def get_logger(
        name: str = "agent",
        console_level: int = logging.INFO, # 普通模式
        file_level: int = logging.DEBUG,
        log_file: str | None = None,
) -> logging.Logger:
    # 初始化 logging 会按照名字管理 Logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 避免重复添加Handler
    if logger.handlers:
        return logger

    # Handler决定日志要输出到哪里，以及哪些级别的日志可以输出，大于等于level的可以输出
    # 控制台的Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)

    logger.addHandler(console_handler)

    # 配置文件Handler
    if not log_file:
        log_file = os.path.join(LOG_ROOT, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)

    logger.addHandler(file_handler)

    return logger

# 快捷获取日志管理器
logger = get_logger()


if __name__ == '__main__':
    logger.info("信息日志")
    logger.error("错误日志")
    logger.warning("警告日志")
    logger.debug("调试日志")