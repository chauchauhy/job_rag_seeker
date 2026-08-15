"""Centralised logging setup.

Console-only logging to stderr, so output is captured by process supervisors
(e.g. Railway container logs). Configure once from an entry point; do not
import this from ``config.py`` so pytest's own logging capture is untouched.
"""

import logging
import os
import sys

__all__ = ["setup_logging", "get_logger"]

DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging() -> None:
    """Configure the root logger once. Level comes from ``LOG_LEVEL``."""
    root = logging.getLogger()
    if getattr(root, "_job_seeker_configured", False):
        return
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)
    root._job_seeker_configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
