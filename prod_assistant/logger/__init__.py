# logger/__init__.py
# Exposes a single shared logger instance (GLOBAL_LOGGER) used across the entire project.
# Import with: from prod_assistant.logger import GLOBAL_LOGGER as log

from .custom_logger import CustomLogger

# Create a single shared logger instance at import time — all modules use this same logger
GLOBAL_LOGGER = CustomLogger().get_logger("prod_assistant")