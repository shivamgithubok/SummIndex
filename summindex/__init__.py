
__version__ = "1.0.0"
__author__ = "SummIndex Team"
__description__ = "Real-time news summarization and indexing system"

from .core.pipeline import SummIndexPipeline
from .ingestion.gnews_client import GNewsClient
from .preprocessing.text_processor import TextProcessor
from .clustering.topic_clustering import TopicClustering
from .summarization.multi_model_summarizer import MultiModelSummarizer
from .embedding.semantic_embedder import SemanticEmbedder
from .indexing.search_index import SearchIndex
from .memory.semantic_memory import SemanticMemory
from .evaluation.metrics import EvaluationMetrics

__all__ = [
    "SummIndexPipeline",
    "GNewsClient",
    "TextProcessor", 
    "TopicClustering",
    "MultiModelSummarizer",
    "SemanticEmbedder",
    "SearchIndex",
    "SemanticMemory",
    "EvaluationMetrics"
]
