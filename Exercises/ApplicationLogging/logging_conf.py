"""File-based logging configuration demo.

Loads a logging config from `logging.conf` (ConfigParser .ini-style format),
then swaps the stdout StreamHandler for a RichHandler so the console output
is colorized.

Run from inside Exercises/ApplicationLogging/:
    python logging_conf.py
"""
import logging
import logging.config
import sys
from pathlib import Path
from rich.logging import RichHandler

# Setup directories
BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = Path(BASE_DIR, "logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Load configuration from file. Pass an absolute path so it works no
# matter where the user invokes Python from.
logging.config.fileConfig(BASE_DIR / "logging.conf")
logger = logging.getLogger()

# Replace the stdout StreamHandler with a RichHandler. RotatingFileHandler
# is a subclass of StreamHandler, so we also check `.stream is sys.stdout`
# to be sure we don't accidentally swap a file handler.
for i, handler in enumerate(logger.handlers):
    if isinstance(handler, logging.StreamHandler) and getattr(handler, "stream", None) is sys.stdout:
        logger.handlers[i] = RichHandler(markup=True)
        break

# Test the logger
logger.debug("Debug message using file-based configuration")
logger.info("Info message using file-based configuration")
logger.warning("Warning message using file-based configuration")
logger.error("Error message using file-based configuration")

