import logging
import sys
from io import StringIO

def setup_logger(level: int = logging.DEBUG) -> tuple[logging.Logger, StringIO]:
    log_stream = StringIO()
    logger = logging.getLogger()
    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    # StringIO handler
    string_handler = logging.StreamHandler(log_stream)
    string_handler.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    string_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(string_handler)

    return logger, log_stream