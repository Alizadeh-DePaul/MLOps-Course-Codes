import logging
from pathlib import Path

# Create logs directory
BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = Path(BASE_DIR, "logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configure file logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=LOGS_DIR / 'app.log',
    filemode='a'  # append mode
)

logger = logging.getLogger(__name__)
logger.debug("This message goes to the log file")
logger.info("Starting data preprocessing pipeline")
logger.warning("Missing values detected in feature X")


