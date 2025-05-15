
import logging
import logging.config
from pathlib import Path
from rich.logging import RichHandler

# Setup directories
BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = Path(BASE_DIR, "logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Load configuration from file
logging.config.fileConfig('logging.conf')
logger = logging.getLogger()

# Replace console handler with Rich handler
for i, handler in enumerate(logger.handlers):
    if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
        logger.handlers[i] = RichHandler(markup=True)
        break

# Test the logger
logger.debug("Debug message using file-based configuration")
logger.info("Info message using file-based configuration")
logger.warning("Warning message using file-based configuration")
logger.error("Error message using file-based configuration")

