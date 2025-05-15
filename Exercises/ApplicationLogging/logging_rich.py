
import logging
import sys
from pathlib import Path
from logging.config import dictConfig
from rich.logging import RichHandler
import time

# Setup directories
BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = Path(BASE_DIR, "logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Advanced logging configuration with Rich
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(message)s"},  # Rich adds its own formatting
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

# Replace the standard console handler with Rich handler
logger.root.handlers[0] = RichHandler(markup=True, rich_tracebacks=True)

# Test all logging levels with more ML-related examples
logger.debug("[bold]Debug:[/bold] Loading dataset with 10,000 samples")
logger.info("[bold green]Info:[/bold green] Model training started with learning rate=0.001")
logger.warning("[bold yellow]Warning:[/bold yellow] Validation loss increasing, possible overfitting")
logger.error("[bold red]Error:[/bold red] Failed to save model weights to S3")
logger.critical("[bold white on red]CRITICAL:[/bold white on red] Out of memory during batch processing")

# Simulate an ML training loop with progress
def train_epoch(epoch, total_epochs):
    logger.info(f"[bold blue]Epoch {epoch}/{total_epochs}[/bold blue]")
    for batch in range(1, 6):
        logger.debug(f"Processing batch {batch}/5")
        time.sleep(0.2)  # Simulate processing time

        # Simulate occasional warnings
        if batch == 3 and epoch % 2 == 0:
            logger.warning(f"Unusually high loss in batch {batch}")

    # Log metrics
    accuracy = 0.7 + (epoch * 0.03)  # Simulated improving accuracy
    loss = 0.5 - (epoch * 0.05)  # Simulated decreasing loss
    logger.info(f"Epoch {epoch} results: accuracy={accuracy:.2f}, loss={loss:.2f}")

# Run a mini training simulation
total_epochs = 3
for epoch in range(1, total_epochs + 1):
    train_epoch(epoch, total_epochs)

# Simulate an error condition
try:
    logger.info("Attempting to load pretrained weights")
    # Simulate an exception
    if total_epochs > 0:  # Always True in this case
        raise FileNotFoundError("Pretrained model file not found")
except Exception as e:
    logger.exception("Failed to load pretrained weights")