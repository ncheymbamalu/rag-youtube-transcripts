"""This module configures the project's logging."""

import sys
from pathlib import Path

from loguru import logger

from rag_youtube_transcripts.config import Config


logger.remove()
logger.add(sys.stderr, colorize=True, level="INFO")

logs_dir: Path = Config.Paths.logs_dir
logs_dir.mkdir(parents=True, exist_ok=True)
LOG_FILE: Path = logs_dir / "file_{time:YYYY_MM_DD_HH_mm_ss}.log"
logger.add(LOG_FILE, retention="2 days", level="DEBUG")
logger = logger.opt(colors=True)
