
import logging
import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.debug("Debug message in Hydra app")
    logger.info("Info message in Hydra app")
    logger.warning("Warning message in Hydra app")
    logger.error("Error message in Hydra app")

    # Your ML workflow here
    logger.info(f"Starting training with parameters: {cfg.model}")

if __name__ == "__main__":
    main()

