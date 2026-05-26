import sys
from loguru import logger

# Remove default handler
logger.remove()

# Add standard console output format
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)

# Optional: Add a file handler if you want to keep logs in a file
# logger.add("logs/app_{time:YYYY-MM-DD}.log", rotation="10 MB", level="DEBUG")

