"""Central logging. Produces the mandatory logs/events.log (rotating, flushed per line)."""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGGER_NAME = "face_tracker"


def setup_logging(log_dir: str, level: str = "INFO") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
    fh = RotatingFileHandler(Path(log_dir) / "events.log", maxBytes=20_000_000,
                             backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def ev(kind: str, level: int = logging.INFO, **fields):
    """Structured event line:  KIND | key=value | key=value"""
    parts = " | ".join(f"{k}={v}" for k, v in fields.items())
    logging.getLogger(LOGGER_NAME).log(level, f"{kind:<20} | {parts}" if parts else kind)
