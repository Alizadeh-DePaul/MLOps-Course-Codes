
import logging
import sys
from pathlib import Path
from logging.config import dictConfig

# Setup directories
BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = Path(BASE_DIR, "logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Advanced logging configuration
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(asctime)s - %(message)s"},
        "detailed": {
            "format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.INFO,
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,  # 10 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "error.log"),
            "maxBytes": 10485760,  # 10 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "root": {
        "handlers": ["console", "info", "error"],
        "level": logging.DEBUG,
        "propagate": True,
    },
}

# Apply the configuration
dictConfig(logging_config)
logger = logging.getLogger(__name__)

# Test all logging levels
logger.debug("Debug message: This won't show in console but will be processed")
logger.info("Info message: Training epoch 1/10 completed, accuracy: 85.7%")
logger.warning("Warning message: GPU utilization is at 95%")
logger.error("Error message: Failed to connect to data source")
logger.critical("Critical message: Model serving API is down!")

# Simulate a function call with logging
def process_batch(batch_id):
    logger.info(f"Processing batch {batch_id}")
    if batch_id % 5 == 0:
        logger.warning(f"Batch {batch_id} contains outliers")
    return True

# Log some function activity
for i in range(1, 11):
    process_batch(i)

