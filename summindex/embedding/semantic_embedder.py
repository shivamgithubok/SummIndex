import logging
import numpy as np
import asyncio
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
import torch
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class SemanticEmbedder:
    def __init__(self, config: Any):
        self.config = config
        self.model: Optional[SentenceTransformer] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    async def initialize(self):
        try:
            logger.info("Initializing semantic embedding model...")
            
            def load_model():
                model = SentenceTransformer(self.config.EMBEDDING_MODEL)
                model.to(self.device)
                return model
                
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(self.executor, load_model)
            
            logger.info(f"Semantic embedding model loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            raise
            
    async def embed_texts(self, 
                         texts: Union[str, List[str]],
                         normalize: bool = True,
                         batch_size: int = 32) -> np.ndarray:
        if not self.model:
            await self.initialize()
            
        try:
            # Handle single text input
            if isinstance(texts, str):
                texts = [texts]
                
            if not texts:
                return np.array([])
                
            # Check cache first
            cached_embeddings = []
            texts_to_embed = []
            cache_indices = []
            
            for i, text in enumerate(texts):
                text_hash = self._get_text_hash(text)
                if text_hash in self.embedding_cache:
                    cached_embeddings.append((i, self.embedding_cache[text_hash]))
                    self.cache_hits += 1
                else:
                    texts_to_embed.append(text)
                    cache_indices.append((i, text_hash))
                    self.cache_misses += 1
                    
            # Generate embeddings for uncached texts
            new_embeddings = []
            if texts_to_embed:
                new_embeddings = await self._generate_embeddings(
                    texts_to_embed, normalize, batch_size
                )
                
                # Cache new embeddings
                for (idx, text_hash), embedding in zip(cache_indices, new_embeddings):
                    self.embedding_cache[text_hash] = embedding
                    
            # Combine cached and new embeddings in correct order
            all_embeddings = [None] * len(texts)
            
            # Place cached embeddings
            for idx, embedding in cached_embeddings:
                all_embeddings[idx] = embedding
                
            # Place new embeddings
            new_idx = 0
            for idx, text_hash in cache_indices:
                all_embeddings[idx] = new_embeddings[new_idx]
                new_idx += 1
                
            result = np.array(all_embeddings)
            
            # Clean cache if it gets too large
            if len(self.embedding_cache) > self.config.CACHE_SIZE:
                await self._cleanup_cache()
                
            return result
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return np.array([])
            
    async def _generate_embeddings(self, 
                                 texts: List[str],
                                 normalize: bool,
                                 batch_size: int) -> List[np.ndarray]:
        
        def generate():
            try:
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    normalize_embeddings=normalize,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                return [emb for emb in embeddings]
                
            except Exception as e:
                logger.error(f"Error in embedding generation: {e}")
                raise
                
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, generate)
        
    async def embed_articles(self, articles: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        try:
            # Extract texts and IDs
            texts = []
            article_ids = []
            
            for article in articles:
                text = self._get_article_text(article)
                if text:
                    texts.append(text)
                    article_ids.append(article.get("id", ""))
                    
            if not texts:
                return {}
                
            # Generate embeddings
            embeddings = await self.embed_texts(texts)
            
            # Create ID to embedding mapping
            embedding_dict = {}
            for article_id, embedding in zip(article_ids, embeddings):
                if article_id:
                    embedding_dict[article_id] = embedding
                    
            return embedding_dict
            
        except Exception as e:
            logger.error(f"Error embedding articles: {e}")
            return {}
            
    async def embed_summaries(self, summaries: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        try:
            texts = []
            summary_ids = []
            
            for summary in summaries:
                text = summary.get("summary", "")
                if text and text != "No summary available":
                    texts.append(text)
                    summary_ids.append(summary.get("cluster_id", ""))
                    
            if not texts:
                return {}
                
            embeddings = await self.embed_texts(texts)
            
            embedding_dict = {}
            for summary_id, embedding in zip(summary_ids, embeddings):
                if summary_id:
                    embedding_dict[summary_id] = embedding
                    
            return embedding_dict
            
        except Exception as e:
            logger.error(f"Error embedding summaries: {e}")
            return {}
            
    def _get_article_text(self, article: Dict[str, Any]) -> str:
        """Extract the best text representation from an article"""
        # Priority: full_text_clean > full_text > title + description
        text = article.get("full_text_clean")
        if text:
            return text
            
        text = article.get("full_text")
        if text:
            return text
            
        # Fallback to title + description
        title = article.get("title", "")
        description = article.get("description", "")
        
        if title and description:
            return f"{title}. {description}"
        elif title:
            return title
        elif description:
            return description
        else:
            return ""
            
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text caching"""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()
        
    async def calculate_similarity(self, 
                                 embedding1: np.ndarray,
                                 embedding2: np.ndarray,
                                 method: str = "cosine") -> float:
        try:
            if method == "cosine":
                # Cosine similarity
                norm1 = np.linalg.norm(embedding1)
                norm2 = np.linalg.norm(embedding2)
                
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                    
                return np.dot(embedding1, embedding2) / (norm1 * norm2)
                
            elif method == "dot":
                # Dot product
                return np.dot(embedding1, embedding2)
                
            elif method == "euclidean":
                # Euclidean distance (converted to similarity)
                distance = np.linalg.norm(embedding1 - embedding2)
                return 1.0 / (1.0 + distance)
                
            else:
                raise ValueError(f"Unknown similarity method: {method}")
                
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
            
    async def find_similar_texts(self, 
                               query_text: str,
                               text_embeddings: Dict[str, np.ndarray],
                               top_k: int = 5,
                               threshold: float = 0.7) -> List[tuple]:
        """
        Find texts similar to query text
        
        Args:
            query_text: Query text
            text_embeddings: Dictionary of text_id -> embedding
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (text_id, similarity_score) tuples
        """
        try:
            if not text_embeddings:
                return []
                
            # Get query embedding
            query_embedding = await self.embed_texts(query_text)
            if query_embedding.size == 0:
                return []
                
            query_embedding = query_embedding[0]
            
            # Calculate similarities
            similarities = []
            for text_id, embedding in text_embeddings.items():
                similarity = await self.calculate_similarity(query_embedding, embedding)
                if similarity >= threshold:
                    similarities.append((text_id, similarity))
                    
            # Sort and return top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar texts: {e}")
            return []
            
    async def cluster_embeddings(self, 
                               embeddings: np.ndarray,
                               n_clusters: Optional[int] = None,
                               method: str = "kmeans") -> np.ndarray:
        """
        Cluster embeddings
        
        Args:
            embeddings: Array of embeddings
            n_clusters: Number of clusters (if None, will be estimated)
            method: Clustering method
            
        Returns:
            Cluster labels
        """
        try:
            if embeddings.size == 0:
                return np.array([])
                
            def cluster():
                if method == "kmeans":
                    from sklearn.cluster import KMeans
                    
                    if n_clusters is None:
                        # Estimate number of clusters
                        n_samples = embeddings.shape[0]
                        estimated_clusters = min(max(2, n_samples // 10), 20)
                    else:
                        estimated_clusters = n_clusters
                        
                    kmeans = KMeans(n_clusters=estimated_clusters, random_state=42)
                    return kmeans.fit_predict(embeddings)
                    
                elif method == "dbscan":
                    from sklearn.cluster import DBSCAN
                    
                    dbscan = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
                    return dbscan.fit_predict(embeddings)
                    
                else:
                    raise ValueError(f"Unknown clustering method: {method}")
                    
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, cluster)
            
        except Exception as e:
            logger.error(f"Error clustering embeddings: {e}")
            return np.array([])
            
    async def _cleanup_cache(self):
        """Clean up embedding cache"""
        # Simple LRU-style cleanup - remove half of the cache
        cache_size = len(self.embedding_cache)
        items_to_remove = cache_size // 2
        
        # Remove oldest items (this is simplified - could be improved with actual LRU)
        cache_items = list(self.embedding_cache.items())
        for key, _ in cache_items[:items_to_remove]:
            del self.embedding_cache[key]
            
        logger.info(f"Cleaned embedding cache: removed {items_to_remove} items, "
                   f"kept {len(self.embedding_cache)} items")
                   
    async def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding statistics"""
        total_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "model_name": self.config.EMBEDDING_MODEL,
            "device": str(self.device),
            "cache_size": len(self.embedding_cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "dimension": self.config.VECTOR_DIMENSION
        }
        
    async def cleanup(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
            
        self.embedding_cache.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Semantic embedder cleaned up")
