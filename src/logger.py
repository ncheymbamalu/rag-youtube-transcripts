"""This module configures the project's logging."""

from datetime import datetime
from pathlib import PosixPath

from loguru import logger

from src.config import Config

logs_dir: PosixPath = Config.logs_dir
logs_dir.mkdir(parents=True, exist_ok=True)
log_file: str = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
logger.add(logs_dir / log_file)
