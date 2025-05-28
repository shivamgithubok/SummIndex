import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from hdbscan import HDBSCAN

logger = logging.getLogger(__name__)

class TopicClustering:
    def __init__(self, config: Any):
        self.config = config
        self.topic_model: Optional[BERTopic] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self.article_clusters: Dict[str, int] = {}  # article_id -> cluster_id
        self.cluster_metadata: Dict[int, Dict[str, Any]] = {}
        self.cluster_update_time: Optional[datetime] = None
        self.topic_evolution: List[Dict[str, Any]] = []
        
    async def initialize(self):
        try:
            logger.info("Initializing topic clustering models...")
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(self.config.CLUSTERING_MODEL)
            
            # Initialize BERTopic with custom configuration
            self.topic_model = BERTopic(
                embedding_model=self.embedding_model,
                min_topic_size=self.config.MIN_CLUSTER_SIZE,
                nr_topics=self.config.MAX_CLUSTERS,
                calculate_probabilities=True,
                verbose=False
            )
            
            logger.info("Topic clustering models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing clustering models: {e}")
            raise
            
    async def cluster_articles(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not articles:
            return {"clusters": {}, "topics": []}
            
        try:
            logger.info(f"Clustering {len(articles)} articles...")
            
            # Extract texts for clustering
            texts = [article.get("full_text_clean", "") for article in articles]
            article_ids = [article.get("id", "") for article in articles]
            
            # Perform clustering
            if len(texts) < self.config.MIN_CLUSTER_SIZE:
                # Not enough articles for clustering
                return await self._handle_small_batch(articles)
                
            # Check if we need to update the model
            if await self._should_update_model():
                topics, probabilities = self.topic_model.fit_transform(texts)
            else:
                topics, probabilities = self.topic_model.transform(texts)
                
            clusters = await self._process_clustering_results(
                articles, topics, probabilities
            )
            
            await self._update_cluster_metadata(clusters)
            
            await self._track_topic_evolution(clusters)
            
            self.cluster_update_time = datetime.utcnow()
            
            logger.info(f"Clustering completed. Found {len(clusters)} clusters")
            
            return {
                "clusters": clusters,
                "topics": self._get_topic_info(),
                "timestamp": self.cluster_update_time
            }
            
        except Exception as e:
            logger.error(f"Error clustering articles: {e}")
            return {"clusters": {}, "topics": []}
            
    async def _should_update_model(self) -> bool:
        """Determine if the topic model should be updated"""
        if not self.cluster_update_time:
            return True
            
        time_since_update = datetime.utcnow() - self.cluster_update_time
        return time_since_update.total_seconds() > self.config.CLUSTER_UPDATE_INTERVAL
        
    async def _handle_small_batch(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle small batches that can't be clustered effectively"""
        # Assign each article to its own cluster
        clusters = {}
        for i, article in enumerate(articles):
            cluster_id = f"single_{i}"
            clusters[cluster_id] = {
                "id": cluster_id,
                "articles": [article],
                "topic": "misc",
                "keywords": self._extract_keywords(article.get("full_text_clean", "")),
                "created_at": datetime.utcnow(),
                "size": 1
            }
            
        return {"clusters": clusters, "topics": []}
        
    async def _process_clustering_results(self, 
                                       articles: List[Dict[str, Any]], 
                                       topics: List[int],
                                       probabilities: np.ndarray) -> Dict[str, Dict[str, Any]]:
        clusters = defaultdict(list)
        
        # Group articles by topic
        for article, topic_id, prob in zip(articles, topics, probabilities):
            if topic_id != -1:  # -1 indicates outlier/noise
                clusters[topic_id].append({
                    "article": article,
                    "probability": float(np.max(prob)) if prob is not None else 0.0
                })
                
        # Convert to structured format
        structured_clusters = {}
        for topic_id, cluster_articles in clusters.items():
            if len(cluster_articles) >= self.config.MIN_CLUSTER_SIZE:
                cluster_info = await self._create_cluster_info(topic_id, cluster_articles)
                structured_clusters[f"topic_{topic_id}"] = cluster_info
                
        return structured_clusters
        
    async def _create_cluster_info(self, 
                                 topic_id: int, 
                                 cluster_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        articles = [item["article"] for item in cluster_articles]
        
        # Get topic keywords from BERTopic
        topic_words = self.topic_model.get_topic(topic_id)
        keywords = [word for word, score in topic_words[:10]] if topic_words else []
        
        # Generate topic label
        topic_label = self._generate_topic_label(keywords, articles)
        
        # Calculate cluster statistics
        avg_probability = np.mean([item["probability"] for item in cluster_articles])
        
        cluster_info = {
            "id": f"topic_{topic_id}",
            "topic_id": topic_id,
            "articles": articles,
            "topic": topic_label,
            "keywords": keywords,
            "size": len(articles),
            "avg_probability": float(avg_probability),
            "created_at": datetime.utcnow(),
            "last_updated": datetime.utcnow(),
            "sources": list(set(article.get("source", "") for article in articles)),
            "time_span": self._calculate_time_span(articles)
        }
        
        return cluster_info
        
    def _generate_topic_label(self, keywords: List[str], articles: List[Dict[str, Any]]) -> str:
        """Generate a human-readable topic label"""
        if not keywords:
            return "general_news"
            
        # Use top keywords to create label
        top_keywords = keywords[:3]
        label = "_".join(top_keywords).lower()
        
        # Clean up the label
        label = "".join(c for c in label if c.isalnum() or c == "_")
        
        return label or "general_news"
        
    def _calculate_time_span(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate time span of articles in cluster"""
        timestamps = []
        for article in articles:
            pub_time = article.get("published_at")
            if pub_time:
                if isinstance(pub_time, str):
                    try:
                        pub_time = datetime.fromisoformat(pub_time.replace('Z', '+00:00'))
                    except:
                        continue
                timestamps.append(pub_time)
                
        if not timestamps:
            return {"start": None, "end": None, "duration_hours": 0}
            
        start_time = min(timestamps)
        end_time = max(timestamps)
        duration = (end_time - start_time).total_seconds() / 3600  # hours
        
        return {
            "start": start_time,
            "end": end_time,
            "duration_hours": duration
        }
        
    async def _update_cluster_metadata(self, clusters: Dict[str, Dict[str, Any]]):
        for cluster_id, cluster_info in clusters.items():
            self.cluster_metadata[cluster_id] = {
                "topic": cluster_info["topic"],
                "keywords": cluster_info["keywords"],
                "size": cluster_info["size"],
                "last_updated": cluster_info["last_updated"],
                "sources": cluster_info["sources"]
            }
            
    async def _track_topic_evolution(self, clusters: Dict[str, Dict[str, Any]]):
        evolution_snapshot = {
            "timestamp": datetime.utcnow(),
            "topics": {},
            "total_clusters": len(clusters)
        }
        
        for cluster_id, cluster_info in clusters.items():
            evolution_snapshot["topics"][cluster_id] = {
                "keywords": cluster_info["keywords"][:5],
                "size": cluster_info["size"],
                "sources": len(cluster_info["sources"])
            }
            
        self.topic_evolution.append(evolution_snapshot)
        
        # Keep only recent evolution data
        cutoff_time = datetime.utcnow() - timedelta(days=7)
        self.topic_evolution = [
            snapshot for snapshot in self.topic_evolution
            if snapshot["timestamp"] > cutoff_time
        ]
        
    def _get_topic_info(self) -> List[Dict[str, Any]]:
        if not self.topic_model:
            return []
            
        topic_info = []
        try:
            for topic_id in self.topic_model.get_topics():
                if topic_id != -1:  # Skip outlier topic
                    topic_words = self.topic_model.get_topic(topic_id)
                    topic_info.append({
                        "id": topic_id,
                        "words": topic_words[:10],
                        "label": self._generate_topic_label(
                            [word for word, score in topic_words[:3]], []
                        )
                    })
        except Exception as e:
            logger.error(f"Error getting topic info: {e}")
            
        return topic_info
        
    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        try:
            if not text or len(text.split()) < 3:
                return []
                
            # Simple TF-IDF based keyword extraction
            vectorizer = TfidfVectorizer(
                max_features=max_keywords,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            
            # Get scores and sort
            scores = tfidf_matrix.toarray()[0]
            keyword_scores = list(zip(feature_names, scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [keyword for keyword, score in keyword_scores[:max_keywords]]
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
            
    async def get_cluster_summary(self, cluster_id: str) -> Optional[Dict[str, Any]]:
        cluster_info = self.cluster_metadata.get(cluster_id)
        if not cluster_info:
            return None
            
        return {
            "id": cluster_id,
            "topic": cluster_info["topic"],
            "keywords": cluster_info["keywords"],
            "size": cluster_info["size"],
            "last_updated": cluster_info["last_updated"],
            "sources": cluster_info["sources"]
        }
        
    async def find_similar_clusters(self, 
                                  text: str, 
                                  threshold: float = 0.7) -> List[str]:
        if not self.embedding_model:
            return []
            
        try:
            # Get embedding for input text
            text_embedding = self.embedding_model.encode([text])
            
            # Find similar clusters (simplified implementation)
            similar_clusters = []
            
            for cluster_id, metadata in self.cluster_metadata.items():
                # Calculate similarity based on keywords (simplified)
                keywords_text = " ".join(metadata["keywords"])
                keyword_embedding = self.embedding_model.encode([keywords_text])
                
                similarity = np.dot(text_embedding[0], keyword_embedding[0]) / (
                    np.linalg.norm(text_embedding[0]) * np.linalg.norm(keyword_embedding[0])
                )
                
                if similarity > threshold:
                    similar_clusters.append(cluster_id)
                    
            return similar_clusters
            
        except Exception as e:
            logger.error(f"Error finding similar clusters: {e}")
            return []
            
    async def cleanup(self):
        # Clean up old cluster data
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        clusters_to_remove = []
        for cluster_id, metadata in self.cluster_metadata.items():
            if metadata["last_updated"] < cutoff_time:
                clusters_to_remove.append(cluster_id)
                
        for cluster_id in clusters_to_remove:
            del self.cluster_metadata[cluster_id]
            
        logger.info(f"Cleaned up {len(clusters_to_remove)} old clusters")
