"""Hydra `job_logging` override demo.

Hydra builds its own logger by calling `logging.config.dictConfig(...)` under
the hood. To customize it, override the `hydra.job_logging` key in your
config -- see `conf/config.yaml`. The same dictConfig schema you'd write
for plain Python (rotating file handlers, RichHandler, etc.) works here.

Run from inside Exercises/ApplicationLogging/:
    python logging_hydra.py
    python logging_hydra.py model.name=resnet50 model.lr=1e-3

Each run lands in outputs/<date>/<time>/ with its own main.log driven by
the override.
"""
import logging

import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.debug("Debug message in Hydra app")
    logger.info("Info message in Hydra app")
    logger.warning("Warning message in Hydra app")
    logger.error("Error message in Hydra app")

    # Pretend ML workflow - just echo the model config back so you can see
    # what Hydra resolved into `cfg.model`.
    logger.info(f"Starting training with parameters: {dict(cfg.model)}")


if __name__ == "__main__":
    main()

