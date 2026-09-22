import logging

# Libraries that log every HTTP request at INFO.
_NOISY_LOGGERS = ("httpx", "elastic_transport", "huggingface_hub", "sentence_transformers")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
