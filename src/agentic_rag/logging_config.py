"""Application-wide logging setup using only the standard library.

Kept deliberately simple (no structlog/loguru dependency) since the project
already has a large dependency surface; stdlib logging with one consistent
format is enough for a CLI + API of this size.
"""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger once with a single, readable stream handler."""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return  # already configured, e.g. re-entered from a test process

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    root_logger.addHandler(handler)
    root_logger.setLevel(level.upper())
