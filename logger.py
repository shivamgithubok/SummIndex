import os
import logging
import sys
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

load_dotenv()

def setup_logger(name: str = "summindex") -> logging.Logger:
    """Setup centralized logger"""
    logger = logging.getLogger(name)
    
    if logger.hasHandlers():
        return logger
    
    logger.setLevel(getattr(logging, os.getenv("LOG_LEVEL", "INFO")))
    
    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.stream.reconfigure(encoding='utf-8')  # Set UTF-8 encoding for console
    
    # File handler with rotation and UTF-8 encoding
    file_handler = RotatingFileHandler(
        os.getenv("LOG_FILE", "summindex.log"),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'  # Set UTF-8 encoding for file
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Global logger instance
logger = setup_logger()