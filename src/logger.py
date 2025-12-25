"""This module configures the project's logging."""

from datetime import datetime
from pathlib import PosixPath

from loguru import logger
from loguru._logger import Logger

from src.config import Config

logger: Logger = logger.opt(colors=True)

logs_dir: PosixPath = Config.logs_dir
logs_dir.mkdir(parents=True, exist_ok=True)
LOG_FILE: str = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
logger.add(logs_dir / LOG_FILE)
