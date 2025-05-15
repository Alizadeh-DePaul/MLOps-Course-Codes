
import logging
from rich.logging import RichHandler
from pathlib import Path

# Setup
BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = Path(BASE_DIR, "logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Get root logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Clear existing handlers (important when re-configuring)
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Create handlers
console_handler = RichHandler(markup=True)
console_handler.setLevel(logging.INFO)

info_handler = logging.handlers.RotatingFileHandler(
    filename=Path(LOGS_DIR, "info.log"),
    maxBytes=10485760,  # 10 MB
    backupCount=10,
)
info_handler.setLevel(logging.INFO)

error_handler = logging.handlers.RotatingFileHandler(
    filename=Path(LOGS_DIR, "error.log"),
    maxBytes=10485760,  # 10 MB
    backupCount=10,
)
error_handler.setLevel(logging.ERROR)

# Create formatters
minimal_formatter = logging.Formatter(fmt="%(message)s")
detailed_formatter = logging.Formatter(
    fmt="%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
)

# Connect formatters to handlers
console_handler.setFormatter(minimal_formatter)
info_handler.setFormatter(detailed_formatter)
error_handler.setFormatter(detailed_formatter)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(info_handler)
logger.addHandler(error_handler)

# Test the logger
logger.debug("Debug message using programmatic configuration")
logger.info("Info message using programmatic configuration")
logger.warning("Warning message using programmatic configuration")
logger.error("Error message using programmatic configuration")