import logging
import asyncio
import json
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import numpy as np

# Elasticsearch
from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import ConnectionError, NotFoundError

# FAISS for vector search
import faiss

logger = logging.getLogger(__name__)

class SearchIndex:    
    def __init__(self, config: Any):
        self.config = config
        self.es_client: Optional[AsyncElasticsearch] = None
        self.vector_index: Optional[faiss.Index] = None
        self.vector_id_map: Dict[int, str] = {}  # FAISS index -> document ID
        self.document_vectors: Dict[str, np.ndarray] = {}
        self.next_vector_id = 0
        
    async def initialize(self):
        try:
            logger.info("Initializing search index...")
            
            # Initialize Elasticsearch
            await self._init_elasticsearch()
            
            # Initialize FAISS vector index
            await self._init_faiss_index()
            
            logger.info("Search index initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing search index: {e}")
            raise
            
    async def _init_elasticsearch(self):
        try:
            # Create Elasticsearch client
            self.es_client = AsyncElasticsearch(
                hosts=[{
                    'host': self.config.ELASTICSEARCH_HOST,
                    'port': self.config.ELASTICSEARCH_PORT
                }],
                timeout=30,
                max_retries=3,
                retry_on_timeout=True
            )
            
            # Test connection
            await self.es_client.ping()
            
            # Create index if it doesn't exist
            await self._create_elasticsearch_index()
            
            logger.info("Elasticsearch initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Elasticsearch: {e}")
            # Continue without Elasticsearch - use vector search only
            self.es_client = None
            
    async def _create_elasticsearch_index(self):
        index_name = self.config.ELASTICSEARCH_INDEX
        
        # Check if index exists
        if await self.es_client.indices.exists(index=index_name):
            logger.info(f"Elasticsearch index '{index_name}' already exists")
            return
            
        # Define index mapping
        mapping = {
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "type": {"type": "keyword"},  # article, summary, cluster
                    "title": {
                        "type": "text",
                        "analyzer": "english",
                        "fields": {"keyword": {"type": "keyword"}}
                    },
                    "content": {
                        "type": "text",
                        "analyzer": "english"
                    },
                    "summary": {
                        "type": "text",
                        "analyzer": "english"
                    },
                    "keywords": {"type": "keyword"},
                    "topic": {"type": "keyword"},
                    "source": {"type": "keyword"},
                    "sources": {"type": "keyword"},
                    "language": {"type": "keyword"},
                    "published_at": {"type": "date"},
                    "created_at": {"type": "date"},
                    "cluster_id": {"type": "keyword"},
                    "article_count": {"type": "integer"},
                    "word_count": {"type": "integer"},
                    "quality_score": {"type": "float"},
                    "model_used": {"type": "keyword"}
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "english": {
                            "type": "english"
                        }
                    }
                }
            }
        }
        
        # Create index
        await self.es_client.indices.create(
            index=index_name,
            body=mapping
        )
        
        logger.info(f"Created Elasticsearch index '{index_name}'")
        
    async def _init_faiss_index(self):
        """Initialize FAISS vector index"""
        try:
            dimension = self.config.VECTOR_DIMENSION
            
            # Create appropriate FAISS index based on configuration
            if self.config.FAISS_INDEX_TYPE == "IndexFlatIP":
                # Inner product index (for normalized vectors)
                self.vector_index = faiss.IndexFlatIP(dimension)
            elif self.config.FAISS_INDEX_TYPE == "IndexFlatL2":
                # L2 distance index
                self.vector_index = faiss.IndexFlatL2(dimension)
            else:
                # Default to inner product
                self.vector_index = faiss.IndexFlatIP(dimension)
                
            logger.info(f"FAISS index initialized with dimension {dimension}")
            
        except Exception as e:
            logger.error(f"Error initializing FAISS index: {e}")
            self.vector_index = None
            
    async def index_articles(self, 
                           articles: List[Dict[str, Any]],
                           embeddings: Optional[Dict[str, np.ndarray]] = None):
        try:
            if not articles:
                return
                
            logger.info(f"Indexing {len(articles)} articles...")
            
            # Prepare documents for Elasticsearch
            es_docs = []
            vector_docs = []
            
            for article in articles:
                # Prepare Elasticsearch document
                es_doc = self._prepare_article_for_es(article)
                es_docs.append(es_doc)
                
                # Prepare vector document if embedding is available
                article_id = article.get("id")
                if embeddings and article_id in embeddings:
                    vector_docs.append({
                        "id": article_id,
                        "embedding": embeddings[article_id]
                    })
                    
            # Index in Elasticsearch
            if self.es_client and es_docs:
                await self._bulk_index_elasticsearch(es_docs, "article")
                
            # Index in FAISS
            if vector_docs:
                await self._index_vectors(vector_docs)
                
            logger.info(f"Successfully indexed {len(articles)} articles")
            
        except Exception as e:
            logger.error(f"Error indexing articles: {e}")
            
    async def index_summaries(self,
                            summaries: List[Dict[str, Any]],
                            embeddings: Optional[Dict[str, np.ndarray]] = None):

        try:
            if not summaries:
                return
                
            logger.info(f"Indexing {len(summaries)} summaries...")
            
            # Prepare documents
            es_docs = []
            vector_docs = []
            
            for summary in summaries:
                # Prepare Elasticsearch document
                es_doc = self._prepare_summary_for_es(summary)
                es_docs.append(es_doc)
                
                # Prepare vector document
                cluster_id = summary.get("cluster_id")
                if embeddings and cluster_id in embeddings:
                    vector_docs.append({
                        "id": cluster_id,
                        "embedding": embeddings[cluster_id]
                    })
                    
            # Index in Elasticsearch
            if self.es_client and es_docs:
                await self._bulk_index_elasticsearch(es_docs, "summary")
                
            # Index in FAISS
            if vector_docs:
                await self._index_vectors(vector_docs)
                
            logger.info(f"Successfully indexed {len(summaries)} summaries")
            
        except Exception as e:
            logger.error(f"Error indexing summaries: {e}")
            
    def _prepare_article_for_es(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare article document for Elasticsearch indexing"""
        return {
            "id": article.get("id"),
            "type": "article",
            "title": article.get("title", ""),
            "content": article.get("full_text_clean", article.get("full_text", "")),
            "source": article.get("source", ""),
            "language": article.get("language_detected", article.get("language", "")),
            "published_at": self._format_datetime(article.get("published_at")),
            "created_at": self._format_datetime(article.get("fetched_at")),
            "word_count": article.get("word_count", 0),
            "url": article.get("url", "")
        }
        
    def _prepare_summary_for_es(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare summary document for Elasticsearch indexing"""
        return {
            "id": summary.get("cluster_id"),
            "type": "summary",
            "title": f"Summary: {summary.get('topic', 'News')}",
            "content": summary.get("summary", ""),
            "summary": summary.get("summary", ""),
            "keywords": summary.get("keywords", []),
            "topic": summary.get("topic", ""),
            "sources": summary.get("sources", []),
            "cluster_id": summary.get("cluster_id"),
            "article_count": summary.get("article_count", 0),
            "word_count": summary.get("word_count", 0),
            "quality_score": summary.get("quality_score", 0.0),
            "model_used": summary.get("model_used", ""),
            "created_at": self._format_datetime(summary.get("created_at"))
        }
        
    def _format_datetime(self, dt: Any) -> Optional[str]:
        """Format datetime for Elasticsearch"""
        if not dt:
            return None
            
        if isinstance(dt, str):
            return dt
        elif isinstance(dt, datetime):
            return dt.isoformat()
        else:
            return None
            
    async def _bulk_index_elasticsearch(self, 
                                      documents: List[Dict[str, Any]],
                                      doc_type: str):
        """Bulk index documents in Elasticsearch"""
        if not self.es_client:
            return
            
        try:
            # Prepare bulk actions
            actions = []
            for doc in documents:
                action = {
                    "_index": self.config.ELASTICSEARCH_INDEX,
                    "_id": doc["id"],
                    "_source": doc
                }
                actions.append({"index": action})
                
            if actions:
                from elasticsearch.helpers import async_bulk
                
                success_count, failed_items = await async_bulk(
                    self.es_client,
                    (action["index"] for action in actions),
                    request_timeout=60
                )
                
                logger.info(f"Elasticsearch bulk index: {success_count} succeeded, "
                           f"{len(failed_items)} failed")
                           
        except Exception as e:
            logger.error(f"Error in Elasticsearch bulk indexing: {e}")
            
    async def _index_vectors(self, vector_docs: List[Dict[str, Any]]):
        """Index vectors in FAISS"""
        if not self.vector_index:
            return
            
        try:
            # Prepare vectors and IDs
            vectors = []
            doc_ids = []
            
            for doc in vector_docs:
                embedding = doc["embedding"]
                if isinstance(embedding, np.ndarray) and embedding.size > 0:
                    vectors.append(embedding)
                    doc_ids.append(doc["id"])
                    
            if not vectors:
                return
                
            # Convert to numpy array
            vector_matrix = np.array(vectors).astype('float32')
            
            # Normalize vectors for inner product search
            if self.config.FAISS_INDEX_TYPE == "IndexFlatIP":
                faiss.normalize_L2(vector_matrix)
                
            # Add vectors to index
            start_id = self.next_vector_id
            self.vector_index.add(vector_matrix)
            
            # Update ID mapping
            for i, doc_id in enumerate(doc_ids):
                faiss_id = start_id + i
                self.vector_id_map[faiss_id] = doc_id
                self.document_vectors[doc_id] = vectors[i]
                
            self.next_vector_id += len(doc_ids)
            
            logger.info(f"Added {len(vectors)} vectors to FAISS index")
            
        except Exception as e:
            logger.error(f"Error indexing vectors in FAISS: {e}")
            
    async def search_text(self, 
                         query: str,
                         doc_type: Optional[str] = None,
                         size: int = 10,
                         filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not self.es_client:
            logger.warning("Elasticsearch not available for text search")
            return []
            
        try:
            # Build search query
            search_body = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["title^2", "content", "summary", "keywords^1.5"],
                                    "type": "best_fields",
                                    "fuzziness": "AUTO"
                                }
                            }
                        ]
                    }
                },
                "size": size,
                "sort": [
                    {"_score": {"order": "desc"}},
                    {"created_at": {"order": "desc"}}
                ]
            }
            
            # Add type filter
            if doc_type:
                search_body["query"]["bool"]["filter"] = [
                    {"term": {"type": doc_type}}
                ]
                
            # Add additional filters
            if filters:
                if "filter" not in search_body["query"]["bool"]:
                    search_body["query"]["bool"]["filter"] = []
                    
                for field, value in filters.items():
                    if isinstance(value, list):
                        search_body["query"]["bool"]["filter"].append(
                            {"terms": {field: value}}
                        )
                    else:
                        search_body["query"]["bool"]["filter"].append(
                            {"term": {field: value}}
                        )
                        
            # Execute search
            response = await self.es_client.search(
                index=self.config.ELASTICSEARCH_INDEX,
                body=search_body
            )
            
            # Process results
            results = []
            for hit in response["hits"]["hits"]:
                result = hit["_source"]
                result["_score"] = hit["_score"]
                results.append(result)
                
            logger.debug(f"Text search for '{query}' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in text search: {e}")
            return []
            
    async def search_vector(self,
                          query_vector: np.ndarray,
                          k: int = 10,
                          threshold: float = 0.7) -> List[Dict[str, Any]]:
        if not self.vector_index:
            logger.warning("FAISS index not available for vector search")
            return []
            
        try:
            # Normalize query vector if using inner product
            if self.config.FAISS_INDEX_TYPE == "IndexFlatIP":
                query_vector = query_vector.copy().astype('float32')
                faiss.normalize_L2(query_vector.reshape(1, -1))
                query_vector = query_vector.flatten()
            else:
                query_vector = query_vector.astype('float32')
                
            # Search
            scores, indices = self.vector_index.search(
                query_vector.reshape(1, -1), k
            )
            
            # Process results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # No more results
                    break
                    
                # Convert score to similarity (depends on index type)
                if self.config.FAISS_INDEX_TYPE == "IndexFlatIP":
                    similarity = float(score)
                else:
                    # For L2, convert distance to similarity
                    similarity = 1.0 / (1.0 + float(score))
                    
                if similarity >= threshold:
                    doc_id = self.vector_id_map.get(idx)
                    if doc_id:
                        results.append({
                            "id": doc_id,
                            "similarity": similarity,
                            "faiss_index": int(idx)
                        })
                        
            logger.debug(f"Vector search returned {len(results)} results above threshold")
            return results
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []
            
    async def hybrid_search(self,
                          query: str,
                          query_vector: Optional[np.ndarray] = None,
                          doc_type: Optional[str] = None,
                          size: int = 10,
                          text_weight: float = 0.7,
                          vector_weight: float = 0.3) -> List[Dict[str, Any]]:
        try:
            # Get text search results
            text_results = await self.search_text(query, doc_type, size * 2)
            
            # Get vector search results if vector provided
            vector_results = []
            if query_vector is not None:
                vector_results = await self.search_vector(query_vector, size * 2)
                
            # Combine results
            combined_scores = {}
            
            # Process text results
            max_text_score = max((r.get("_score", 0) for r in text_results), default=1.0)
            for result in text_results:
                doc_id = result["id"]
                normalized_score = (result.get("_score", 0) / max_text_score) * text_weight
                combined_scores[doc_id] = {
                    "score": normalized_score,
                    "data": result,
                    "text_score": result.get("_score", 0)
                }
                
            # Process vector results
            for result in vector_results:
                doc_id = result["id"]
                vector_score = result["similarity"] * vector_weight
                
                if doc_id in combined_scores:
                    combined_scores[doc_id]["score"] += vector_score
                    combined_scores[doc_id]["vector_score"] = result["similarity"]
                else:
                    # Need to get document data (simplified - could fetch from ES)
                    combined_scores[doc_id] = {
                        "score": vector_score,
                        "data": {"id": doc_id, "type": "unknown"},
                        "vector_score": result["similarity"]
                    }
                    
            # Sort by combined score and return top results
            sorted_results = sorted(
                combined_scores.values(),
                key=lambda x: x["score"],
                reverse=True
            )
            
            final_results = []
            for item in sorted_results[:size]:
                result = item["data"].copy()
                result["combined_score"] = item["score"]
                if "text_score" in item:
                    result["text_score"] = item["text_score"]
                if "vector_score" in item:
                    result["vector_score"] = item["vector_score"]
                final_results.append(result)
                
            return final_results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
            
    async def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID from Elasticsearch"""
        if not self.es_client:
            return None
            
        try:
            response = await self.es_client.get(
                index=self.config.ELASTICSEARCH_INDEX,
                id=doc_id
            )
            return response["_source"]
            
        except NotFoundError:
            return None
        except Exception as e:
            logger.error(f"Error getting document {doc_id}: {e}")
            return None
            
    async def delete_documents(self, doc_ids: List[str]):
        """Delete documents from indices"""
        try:
            # Delete from Elasticsearch
            if self.es_client:
                for doc_id in doc_ids:
                    try:
                        await self.es_client.delete(
                            index=self.config.ELASTICSEARCH_INDEX,
                            id=doc_id
                        )
                    except NotFoundError:
                        pass  # Document doesn't exist
                        
            # Delete from FAISS (more complex - would need to rebuild index)
            # For now, just remove from tracking
            for doc_id in doc_ids:
                if doc_id in self.document_vectors:
                    del self.document_vectors[doc_id]
                    
            logger.info(f"Deleted {len(doc_ids)} documents from indices")
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            
    async def get_index_statistics(self) -> Dict[str, Any]:
        """Get statistics about the search index"""
        stats = {
            "elasticsearch": {},
            "faiss": {},
            "total_documents": 0
        }
        
        try:
            # Elasticsearch stats
            if self.es_client:
                es_stats = await self.es_client.indices.stats(
                    index=self.config.ELASTICSEARCH_INDEX
                )
                
                index_stats = es_stats["indices"].get(self.config.ELASTICSEARCH_INDEX, {})
                stats["elasticsearch"] = {
                    "total_docs": index_stats.get("total", {}).get("docs", {}).get("count", 0),
                    "size_bytes": index_stats.get("total", {}).get("store", {}).get("size_in_bytes", 0)
                }
                
            # FAISS stats
            if self.vector_index:
                stats["faiss"] = {
                    "total_vectors": self.vector_index.ntotal,
                    "dimension": self.vector_index.d,
                    "index_type": type(self.vector_index).__name__
                }
                
            stats["total_documents"] = len(self.document_vectors)
            
        except Exception as e:
            logger.error(f"Error getting index statistics: {e}")
            
        return stats
        
    async def cleanup(self):
        """Cleanup search index resources"""
        try:
            if self.es_client:
                await self.es_client.close()
                
            # Clear FAISS data
            self.vector_index = None
            self.vector_id_map.clear()
            self.document_vectors.clear()
            
            logger.info("Search index cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up search index: {e}")
