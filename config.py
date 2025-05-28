import os
from typing import List, Dict, Any
from dataclasses import dataclass, field

@dataclass
class Config:
    """Configuration class for SummIndex system"""
    
    # API Configuration
    GNEWS_API_KEY: str = os.getenv("GNEWS_API_KEY", "default_gnews_key")
    API_PORT: int = int(os.getenv("API_PORT", "5000"))  # Changed to 5000 for Replit
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    
    # Elasticsearch Configuration
    ELASTICSEARCH_HOST: str = os.getenv("ELASTICSEARCH_HOST", "localhost")
    ELASTICSEARCH_PORT: int = int(os.getenv("ELASTICSEARCH_PORT", "9200"))
    ELASTICSEARCH_INDEX: str = os.getenv("ELASTICSEARCH_INDEX", "summindex")
    
    # Model Configuration
    SUMMARIZATION_MODELS: Dict[str, str] = field(default_factory=lambda: {
        "pegasus": "google/pegasus-cnn_dailymail",
        "bart": "facebook/bart-large-cnn",
        "t5": "t5-base",
        "longformer": "allenai/led-base-16384"
    })
    
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
    CLUSTERING_MODEL: str = os.getenv("CLUSTERING_MODEL", "all-MiniLM-L6-v2")
    
    # Processing Configuration
    MAX_ARTICLES_PER_BATCH: int = int(os.getenv("MAX_ARTICLES_PER_BATCH", "50"))
    PROCESSING_INTERVAL: int = int(os.getenv("PROCESSING_INTERVAL", "30"))  # seconds
    MAX_SUMMARY_LENGTH: int = int(os.getenv("MAX_SUMMARY_LENGTH", "150"))
    MIN_SUMMARY_LENGTH: int = int(os.getenv("MIN_SUMMARY_LENGTH", "50"))
    
    # Memory Configuration
    SEMANTIC_MEMORY_SIZE: int = int(os.getenv("SEMANTIC_MEMORY_SIZE", "10000"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.8"))
    
    # Clustering Configuration
    MIN_CLUSTER_SIZE: int = int(os.getenv("MIN_CLUSTER_SIZE", "3"))
    MAX_CLUSTERS: int = int(os.getenv("MAX_CLUSTERS", "50"))
    CLUSTER_UPDATE_INTERVAL: int = int(os.getenv("CLUSTER_UPDATE_INTERVAL", "300"))  # seconds
    
    # Performance Targets
    TARGET_LATENCY: float = float(os.getenv("TARGET_LATENCY", "2.0"))  # seconds
    TARGET_ACCURACY: float = float(os.getenv("TARGET_ACCURACY", "0.94"))
    
    # News Sources Configuration
    NEWS_SOURCES: List[str] = field(default_factory=lambda: [
        "cnn", "bbc-news", "reuters", "associated-press", 
        "the-new-york-times", "the-washington-post", "fox-news"
    ])
    
    NEWS_LANGUAGES: List[str] = field(default_factory=lambda: ["en"])
    NEWS_COUNTRIES: List[str] = field(default_factory=lambda: ["us", "gb", "ca", "au"])
    
    # FAISS Configuration
    FAISS_INDEX_TYPE: str = os.getenv("FAISS_INDEX_TYPE", "IndexFlatIP")
    VECTOR_DIMENSION: int = int(os.getenv("VECTOR_DIMENSION", "768"))
    
    # Reinforcement Learning Configuration
    RL_MODEL_PATH: str = os.getenv("RL_MODEL_PATH", "./models/rl_agent")
    RL_TRAINING_EPISODES: int = int(os.getenv("RL_TRAINING_EPISODES", "1000"))
    RL_UPDATE_FREQUENCY: int = int(os.getenv("RL_UPDATE_FREQUENCY", "100"))
    
    # Evaluation Configuration
    EVALUATION_INTERVAL: int = int(os.getenv("EVALUATION_INTERVAL", "600"))  # seconds
    ROUGE_TYPES: List[str] = field(default_factory=lambda: ["rouge1", "rouge2", "rougeL"])
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "summindex.log")
    
    # Cache Configuration
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))  # seconds
    CACHE_SIZE: int = int(os.getenv("CACHE_SIZE", "1000"))
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        if not self.GNEWS_API_KEY or self.GNEWS_API_KEY == "default_gnews_key":
            raise ValueError("GNEWS_API_KEY must be set in environment variables")
            
        if self.TARGET_LATENCY <= 0:
            raise ValueError("TARGET_LATENCY must be positive")
            
        if not 0 < self.TARGET_ACCURACY <= 1:
            raise ValueError("TARGET_ACCURACY must be between 0 and 1")
            
        if self.MAX_SUMMARY_LENGTH <= self.MIN_SUMMARY_LENGTH:
            raise ValueError("MAX_SUMMARY_LENGTH must be greater than MIN_SUMMARY_LENGTH")
            
        return True
