import logging
import os
from dotenv import load_dotenv

load_dotenv()

class SummIndexLogger:
    """Centralized logging configuration"""
    
    _instance = None
    _configured = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._configured:
            self._setup_logging()
            self._configured = True
    
    def _setup_logging(self):
        """Setup centralized logging configuration"""
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        log_file = os.getenv("LOG_FILE", "summindex.log")
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Get logger instance with consistent configuration"""
        SummIndexLogger()  # Ensure logging is configured
        return logging.getLogger(name)

# Global logger getter
def get_logger(name: str) -> logging.Logger:
    """Get centralized logger"""
    return SummIndexLogger.get_logger(name)