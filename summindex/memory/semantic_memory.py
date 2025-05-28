import logging
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict
import faiss
import pickle
import os

logger = logging.getLogger(__name__)

class SemanticMemory:    
    def __init__(self, config: Any):
        self.config = config
        self.memory_index: Optional[faiss.Index] = None
        self.memory_metadata: Dict[int, Dict[str, Any]] = {}
        self.topic_memories: Dict[str, List[int]] = defaultdict(list)
        self.temporal_index: Dict[datetime, List[int]] = defaultdict(list)
        self.summary_embeddings: Dict[str, np.ndarray] = {}
        self.redundancy_threshold = self.config.SIMILARITY_THRESHOLD
        self.next_memory_id = 0
        self.memory_size = 0
        
        # Memory decay parameters
        self.decay_factor = 0.95  # How much importance decays over time
        self.max_age_hours = 24  # Maximum age for memories
        
    async def initialize(self):
        try:
            logger.info("Initializing semantic memory system...")
            
            # Initialize FAISS index for memory vectors
            dimension = self.config.VECTOR_DIMENSION
            self.memory_index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors
            
            # Load existing memory if available
            await self._load_memory_state()
            
            logger.info(f"Semantic memory initialized with {self.memory_size} memories")
            
        except Exception as e:
            logger.error(f"Error initializing semantic memory: {e}")
            raise
            
    async def store_summary_memory(self, 
                                 summary: Dict[str, Any],
                                 embedding: np.ndarray) -> int:
        try:
            if not self.memory_index:
                await self.initialize()
                
            # Normalize embedding for inner product search
            embedding = embedding.copy().astype('float32')
            faiss.normalize_L2(embedding.reshape(1, -1))
            embedding = embedding.flatten()
            
            # Create memory metadata
            memory_id = self.next_memory_id
            self.next_memory_id += 1
            
            memory_metadata = {
                "id": memory_id,
                "cluster_id": summary.get("cluster_id"),
                "topic": summary.get("topic", "unknown"),
                "summary_text": summary.get("summary", ""),
                "keywords": summary.get("keywords", []),
                "sources": summary.get("sources", []),
                "article_count": summary.get("article_count", 0),
                "quality_score": summary.get("quality_score", 0.0),
                "created_at": datetime.utcnow(),
                "access_count": 0,
                "importance": 1.0,
                "embedding": embedding
            }
            
            # Add to FAISS index
            self.memory_index.add(embedding.reshape(1, -1))
            
            # Store metadata
            self.memory_metadata[memory_id] = memory_metadata
            
            # Index by topic
            topic = memory_metadata["topic"]
            self.topic_memories[topic].append(memory_id)
            
            # Index by time
            created_at = memory_metadata["created_at"]
            time_key = created_at.replace(minute=0, second=0, microsecond=0)  # Hour granularity
            self.temporal_index[time_key].append(memory_id)
            
            # Store embedding for quick access
            cluster_id = summary.get("cluster_id")
            if cluster_id:
                self.summary_embeddings[cluster_id] = embedding
                
            self.memory_size += 1
            
            # Clean up old memories if limit exceeded
            if self.memory_size > self.config.SEMANTIC_MEMORY_SIZE:
                await self._cleanup_old_memories()
                
            logger.debug(f"Stored memory {memory_id} for topic '{topic}'")
            return memory_id
            
        except Exception as e:
            logger.error(f"Error storing summary memory: {e}")
            return -1
            
    async def check_redundancy(self, 
                             summary_text: str,
                             embedding: np.ndarray,
                             topic: Optional[str] = None) -> Dict[str, Any]:
        try:
            if not self.memory_index or self.memory_size == 0:
                return {
                    "is_redundant": False,
                    "similarity_score": 0.0,
                    "similar_memories": []
                }
                
            # Normalize embedding
            embedding = embedding.copy().astype('float32')
            faiss.normalize_L2(embedding.reshape(1, -1))
            
            # Search for similar memories
            k = min(10, self.memory_size)  # Check top 10 similar memories
            similarities, indices = self.memory_index.search(embedding.reshape(1, -1), k)
            
            similar_memories = []
            max_similarity = 0.0
            
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx == -1:  # No more results
                    break
                    
                memory = self.memory_metadata.get(idx)
                if not memory:
                    continue
                    
                # Apply topic filter if specified
                if topic and memory["topic"] != topic:
                    continue
                    
                # Check temporal relevance (decay old memories)
                age_hours = (datetime.utcnow() - memory["created_at"]).total_seconds() / 3600
                time_decay = self.decay_factor ** age_hours
                adjusted_similarity = float(similarity) * time_decay
                
                if adjusted_similarity > max_similarity:
                    max_similarity = adjusted_similarity
                    
                if adjusted_similarity > self.redundancy_threshold:
                    similar_memories.append({
                        "memory_id": idx,
                        "similarity": float(similarity),
                        "adjusted_similarity": adjusted_similarity,
                        "topic": memory["topic"],
                        "summary": memory["summary_text"][:100] + "...",
                        "age_hours": age_hours
                    })
                    
            # Sort by adjusted similarity
            similar_memories.sort(key=lambda x: x["adjusted_similarity"], reverse=True)
            
            is_redundant = max_similarity > self.redundancy_threshold
            
            return {
                "is_redundant": is_redundant,
                "similarity_score": max_similarity,
                "similar_memories": similar_memories[:3],  # Top 3 most similar
                "redundancy_threshold": self.redundancy_threshold
            }
            
        except Exception as e:
            logger.error(f"Error checking redundancy: {e}")
            return {
                "is_redundant": False,
                "similarity_score": 0.0,
                "similar_memories": []
            }
            
    async def retrieve_context(self, 
                             topic: str,
                             time_window_hours: int = 24,
                             max_memories: int = 5) -> List[Dict[str, Any]]:
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
            context_memories = []
            
            # Get memories for the topic
            topic_memory_ids = self.topic_memories.get(topic, [])
            
            for memory_id in topic_memory_ids:
                memory = self.memory_metadata.get(memory_id)
                if not memory:
                    continue
                    
                # Check if within time window
                if memory["created_at"] < cutoff_time:
                    continue
                    
                # Calculate relevance score
                age_hours = (datetime.utcnow() - memory["created_at"]).total_seconds() / 3600
                time_decay = self.decay_factor ** age_hours
                relevance_score = memory["importance"] * time_decay * memory["quality_score"]
                
                context_memory = {
                    "memory_id": memory_id,
                    "topic": memory["topic"],
                    "summary": memory["summary_text"],
                    "keywords": memory["keywords"],
                    "created_at": memory["created_at"],
                    "relevance_score": relevance_score,
                    "quality_score": memory["quality_score"]
                }
                
                context_memories.append(context_memory)
                
            # Sort by relevance and return top memories
            context_memories.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            # Update access counts
            for memory in context_memories[:max_memories]:
                memory_id = memory["memory_id"]
                if memory_id in self.memory_metadata:
                    self.memory_metadata[memory_id]["access_count"] += 1
                    
            return context_memories[:max_memories]
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
            
    async def update_memory_importance(self, 
                                     memory_id: int,
                                     feedback: Dict[str, Any]):
        try:
            if memory_id not in self.memory_metadata:
                return
                
            memory = self.memory_metadata[memory_id]
            
            # Update importance based on feedback
            user_rating = feedback.get("user_rating", 0.5)
            usage_frequency = feedback.get("usage_frequency", 0.0)
            quality_feedback = feedback.get("quality_feedback", 0.5)
            
            # Weighted combination
            new_importance = (
                memory["importance"] * 0.6 +
                user_rating * 0.2 +
                usage_frequency * 0.1 +
                quality_feedback * 0.1
            )
            
            memory["importance"] = max(0.1, min(2.0, new_importance))  # Clamp between 0.1 and 2.0
            
            logger.debug(f"Updated memory {memory_id} importance to {memory['importance']:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating memory importance: {e}")
            
    async def find_similar_topics(self, 
                                topic: str,
                                similarity_threshold: float = 0.7) -> List[str]:
        try:
            if topic not in self.topic_memories:
                return []
                
            # Get a representative embedding for the topic
            topic_memory_ids = self.topic_memories[topic]
            if not topic_memory_ids:
                return []
                
            # Average embeddings for the topic
            topic_embeddings = []
            for memory_id in topic_memory_ids:
                memory = self.memory_metadata.get(memory_id)
                if memory and "embedding" in memory:
                    topic_embeddings.append(memory["embedding"])
                    
            if not topic_embeddings:
                return []
                
            topic_embedding = np.mean(topic_embeddings, axis=0).astype('float32')
            faiss.normalize_L2(topic_embedding.reshape(1, -1))
            
            # Compare with other topics
            similar_topics = []
            
            for other_topic, other_memory_ids in self.topic_memories.items():
                if other_topic == topic or not other_memory_ids:
                    continue
                    
                # Get representative embedding for other topic
                other_embeddings = []
                for memory_id in other_memory_ids:
                    memory = self.memory_metadata.get(memory_id)
                    if memory and "embedding" in memory:
                        other_embeddings.append(memory["embedding"])
                        
                if not other_embeddings:
                    continue
                    
                other_embedding = np.mean(other_embeddings, axis=0).astype('float32')
                faiss.normalize_L2(other_embedding.reshape(1, -1))
                
                # Calculate similarity
                similarity = np.dot(topic_embedding, other_embedding)
                
                if similarity >= similarity_threshold:
                    similar_topics.append((other_topic, float(similarity)))
                    
            # Sort by similarity
            similar_topics.sort(key=lambda x: x[1], reverse=True)
            
            return [topic for topic, _ in similar_topics]
            
        except Exception as e:
            logger.error(f"Error finding similar topics: {e}")
            return []
            
    async def get_memory_statistics(self) -> Dict[str, Any]:
        """Get semantic memory statistics"""
        try:
            current_time = datetime.utcnow()
            
            # Basic statistics
            stats = {
                "total_memories": self.memory_size,
                "unique_topics": len(self.topic_memories),
                "memory_limit": self.config.SEMANTIC_MEMORY_SIZE,
                "memory_utilization": self.memory_size / self.config.SEMANTIC_MEMORY_SIZE
            }
            
            # Age distribution
            age_buckets = {"recent": 0, "medium": 0, "old": 0}
            importance_sum = 0.0
            access_sum = 0
            
            for memory in self.memory_metadata.values():
                age_hours = (current_time - memory["created_at"]).total_seconds() / 3600
                
                if age_hours < 6:
                    age_buckets["recent"] += 1
                elif age_hours < 24:
                    age_buckets["medium"] += 1
                else:
                    age_buckets["old"] += 1
                    
                importance_sum += memory["importance"]
                access_sum += memory["access_count"]
                
            stats["age_distribution"] = age_buckets
            
            if self.memory_size > 0:
                stats["average_importance"] = importance_sum / self.memory_size
                stats["average_access_count"] = access_sum / self.memory_size
            else:
                stats["average_importance"] = 0.0
                stats["average_access_count"] = 0.0
                
            # Topic distribution
            topic_counts = {topic: len(memories) for topic, memories in self.topic_memories.items()}
            stats["topic_distribution"] = dict(sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10])
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting memory statistics: {e}")
            return {}
            
    async def _cleanup_old_memories(self):
        """Clean up old or low-importance memories"""
        try:
            if self.memory_size <= self.config.SEMANTIC_MEMORY_SIZE:
                return
                
            current_time = datetime.utcnow()
            memories_to_remove = []
            
            # Calculate removal scores (lower = more likely to be removed)
            memory_scores = []
            for memory_id, memory in self.memory_metadata.items():
                age_hours = (current_time - memory["created_at"]).total_seconds() / 3600
                
                # Skip very recent memories
                if age_hours < 1:
                    continue
                    
                # Calculate removal score
                age_penalty = age_hours / self.max_age_hours
                importance_bonus = memory["importance"]
                access_bonus = min(1.0, memory["access_count"] / 10.0)
                quality_bonus = memory["quality_score"]
                
                removal_score = age_penalty - (importance_bonus + access_bonus + quality_bonus) / 3.0
                memory_scores.append((memory_id, removal_score))
                
            # Sort by removal score (highest scores first for removal)
            memory_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Remove excess memories
            excess_count = self.memory_size - self.config.SEMANTIC_MEMORY_SIZE + 100  # Remove a buffer
            memories_to_remove = [memory_id for memory_id, _ in memory_scores[:excess_count]]
            
            # Remove memories
            for memory_id in memories_to_remove:
                await self._remove_memory(memory_id)
                
            logger.info(f"Cleaned up {len(memories_to_remove)} old memories")
            
        except Exception as e:
            logger.error(f"Error cleaning up old memories: {e}")
            
    async def _remove_memory(self, memory_id: int):
        """Remove a specific memory"""
        try:
            if memory_id not in self.memory_metadata:
                return
                
            memory = self.memory_metadata[memory_id]
            
            # Remove from topic index
            topic = memory["topic"]
            if topic in self.topic_memories:
                if memory_id in self.topic_memories[topic]:
                    self.topic_memories[topic].remove(memory_id)
                if not self.topic_memories[topic]:
                    del self.topic_memories[topic]
                    
            # Remove from temporal index
            created_at = memory["created_at"]
            time_key = created_at.replace(minute=0, second=0, microsecond=0)
            if time_key in self.temporal_index:
                if memory_id in self.temporal_index[time_key]:
                    self.temporal_index[time_key].remove(memory_id)
                if not self.temporal_index[time_key]:
                    del self.temporal_index[time_key]
                    
            # Remove from embedding cache
            cluster_id = memory.get("cluster_id")
            if cluster_id and cluster_id in self.summary_embeddings:
                del self.summary_embeddings[cluster_id]
                
            # Remove metadata
            del self.memory_metadata[memory_id]
            self.memory_size -= 1
            
            # Note: FAISS index doesn't support individual removal efficiently
            # In a production system, you might periodically rebuild the index
            
        except Exception as e:
            logger.error(f"Error removing memory {memory_id}: {e}")
            
    async def _save_memory_state(self):
        """Save memory state to disk"""
        try:
            memory_state = {
                "metadata": self.memory_metadata,
                "topic_memories": dict(self.topic_memories),
                "temporal_index": {str(k): v for k, v in self.temporal_index.items()},
                "summary_embeddings": self.summary_embeddings,
                "next_memory_id": self.next_memory_id,
                "memory_size": self.memory_size
            }
            
            # Save to file
            memory_file = "semantic_memory_state.pkl"
            with open(memory_file, 'wb') as f:
                pickle.dump(memory_state, f)
                
            logger.info("Semantic memory state saved")
            
        except Exception as e:
            logger.error(f"Error saving memory state: {e}")
            
    async def _load_memory_state(self):
        """Load memory state from disk"""
        try:
            memory_file = "semantic_memory_state.pkl"
            if not os.path.exists(memory_file):
                return
                
            with open(memory_file, 'rb') as f:
                memory_state = pickle.load(f)
                
            self.memory_metadata = memory_state.get("metadata", {})
            self.topic_memories = defaultdict(list, memory_state.get("topic_memories", {}))
            
            # Convert temporal index keys back to datetime
            temporal_data = memory_state.get("temporal_index", {})
            for time_str, memory_ids in temporal_data.items():
                time_key = datetime.fromisoformat(time_str)
                self.temporal_index[time_key] = memory_ids
                
            self.summary_embeddings = memory_state.get("summary_embeddings", {})
            self.next_memory_id = memory_state.get("next_memory_id", 0)
            self.memory_size = memory_state.get("memory_size", 0)
            
            # Rebuild FAISS index
            if self.memory_metadata:
                embeddings = []
                for memory in self.memory_metadata.values():
                    if "embedding" in memory:
                        embeddings.append(memory["embedding"])
                        
                if embeddings:
                    embeddings_array = np.array(embeddings).astype('float32')
                    self.memory_index.add(embeddings_array)
                    
            logger.info(f"Loaded semantic memory state with {self.memory_size} memories")
            
        except Exception as e:
            logger.error(f"Error loading memory state: {e}")
            
    async def cleanup(self):
        """Cleanup semantic memory resources"""
        try:
            # Save current state
            await self._save_memory_state()
            
            # Clear in-memory structures
            self.memory_metadata.clear()
            self.topic_memories.clear()
            self.temporal_index.clear()
            self.summary_embeddings.clear()
            
            self.memory_index = None
            self.memory_size = 0
            
            logger.info("Semantic memory cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during semantic memory cleanup: {e}")
