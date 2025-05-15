import logging
import sys
# Create super basic logger
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__) 
# # Logging levels (from lowest to highest priority)
logger.debug("Debug")
logger.info("Information")
logger.warning("Warning")
logger.error("Error")
logger.critical("Critical Error")