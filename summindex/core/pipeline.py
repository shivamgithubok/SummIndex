import logging
import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict

from ..ingestion.gnews_client import GNewsClient
from ..preprocessing.text_processor import TextProcessor
from ..clustering.topic_clustering import TopicClustering
from ..summarization.multi_model_summarizer import MultiModelSummarizer
from ..summarization.incremental_updater import IncrementalUpdater
from ..embedding.semantic_embedder import SemanticEmbedder
from ..indexing.search_index import SearchIndex
from ..memory.semantic_memory import SemanticMemory
from ..rl_agent.summarization_agent import SummarizationAgent
from ..evaluation.metrics import EvaluationMetrics

logger = logging.getLogger(__name__)

class SummIndexPipeline:
    def __init__(self, config: Any):
        self.config = config
        self.running = False
        
        # Core components
        self.gnews_client: Optional[GNewsClient] = None
        self.text_processor: Optional[TextProcessor] = None
        self.topic_clustering: Optional[TopicClustering] = None
        self.summarizer: Optional[MultiModelSummarizer] = None
        self.incremental_updater: Optional[IncrementalUpdater] = None
        self.embedder: Optional[SemanticEmbedder] = None
        self.search_index: Optional[SearchIndex] = None
        self.semantic_memory: Optional[SemanticMemory] = None
        self.rl_agent: Optional[SummarizationAgent] = None
        self.evaluation_metrics: Optional[EvaluationMetrics] = None
        
        # Pipeline state
        self.last_processing_time: Optional[datetime] = None
        self.processing_stats: Dict[str, Any] = defaultdict(list)
        self.current_summaries: Dict[str, Dict[str, Any]] = {}
        self.pipeline_metrics: Dict[str, float] = {}
        
    async def initialize(self):
        """Initialize all pipeline components"""
        try:
            logger.info("Initializing SummIndex pipeline...")
            start_time = time.time()
            
            # Initialize components in dependency order
            
            # 1. Basic processing components
            self.text_processor = TextProcessor(self.config)
            await self.text_processor.initialize() if hasattr(self.text_processor, 'initialize') else None
            
            self.embedder = SemanticEmbedder(self.config)
            await self.embedder.initialize()
            
            # 2. News ingestion
            self.gnews_client = GNewsClient(self.config.GNEWS_API_KEY, self.config)
            await self.gnews_client.initialize()
            
            # 3. Clustering and memory
            self.topic_clustering = TopicClustering(self.config)
            await self.topic_clustering.initialize()
            
            self.semantic_memory = SemanticMemory(self.config)
            await self.semantic_memory.initialize()
            
            # 4. Summarization components
            self.summarizer = MultiModelSummarizer(self.config)
            await self.summarizer.initialize()
            
            self.incremental_updater = IncrementalUpdater(self.config)
            await self.incremental_updater.initialize()
            
            # 5. Indexing and search
            self.search_index = SearchIndex(self.config)
            await self.search_index.initialize()
            
            # 6. RL agent
            self.rl_agent = SummarizationAgent(self.config)
            await self.rl_agent.initialize()
            
            # 7. Evaluation metrics
            self.evaluation_metrics = EvaluationMetrics(self.config)
            await self.evaluation_metrics.initialize()
            
            initialization_time = time.time() - start_time
            logger.info(f"SummIndex pipeline initialized in {initialization_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error initializing pipeline: {e}")
            raise
            
    async def run(self):
        """Main pipeline execution loop"""
        try:
            logger.info("Starting SummIndex pipeline...")
            self.running = True
            
            while self.running:
                try:
                    # Execute one processing cycle
                    await self._process_cycle()
                    
                    # Wait for next cycle
                    await asyncio.sleep(self.config.PROCESSING_INTERVAL)
                    
                except KeyboardInterrupt:
                    logger.info("Received interrupt signal")
                    break
                except Exception as e:
                    logger.error(f"Error in processing cycle: {e}")
                    # Continue running despite errors
                    await asyncio.sleep(10)  # Wait before retrying
                    
        except Exception as e:
            logger.error(f"Fatal error in pipeline: {e}")
        finally:
            self.running = False
            
    async def _process_cycle(self):
        """Execute one complete processing cycle"""
        cycle_start_time = time.time()
        logger.info("Starting processing cycle...")
        
        try:
            # Step 1: Fetch new articles
            articles = await self._fetch_articles()
            if not articles:
                logger.debug("No new articles to process")
                return
                
            # Step 2: Process articles
            processed_articles = await self._process_articles(articles)
            if not processed_articles:
                logger.debug("No articles passed processing")
                return
                
            # Step 3: Generate embeddings
            article_embeddings = await self._generate_article_embeddings(processed_articles)
            
            # Step 4: Cluster articles
            clustering_results = await self._cluster_articles(processed_articles)
            
            # Step 5: RL agent decision
            rl_decision = await self._get_rl_decision(clustering_results)
            
            # Step 6: Execute summarization based on decision
            summaries = await self._execute_summarization(clustering_results, rl_decision)
            
            # Step 7: Update semantic memory and check redundancy
            await self._update_memory_and_check_redundancy(summaries)
            
            # Step 8: Index content
            await self._index_content(processed_articles, summaries, article_embeddings)
            
            # Step 9: Evaluate performance
            await self._evaluate_cycle_performance(cycle_start_time, summaries)
            
            # Step 10: Learn from feedback
            await self._collect_and_process_feedback(summaries, rl_decision)
            
            cycle_time = time.time() - cycle_start_time
            logger.info(f"Processing cycle completed in {cycle_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error in processing cycle: {e}")
            cycle_time = time.time() - cycle_start_time
            await self.evaluation_metrics.measure_latency("failed_cycle", cycle_start_time)
            
    async def _fetch_articles(self) -> List[Dict[str, Any]]:
        """Fetch new articles from news sources"""
        try:
            fetch_start_time = time.time()
            
            # Fetch articles
            articles = await self.gnews_client.fetch_news(
                max_articles=self.config.MAX_ARTICLES_PER_BATCH
            )
            
            # Measure latency
            await self.evaluation_metrics.measure_latency("fetch_articles", fetch_start_time)
            
            logger.info(f"Fetched {len(articles)} articles")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching articles: {e}")
            return []
            
    async def _process_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and clean articles"""
        try:
            process_start_time = time.time()
            
            # Process articles
            processed_articles = await self.text_processor.process_articles(articles)
            
            # Filter valid articles
            valid_articles = [
                article for article in processed_articles
                if self.text_processor.validate_article_quality(article)
            ]
            
            # Measure latency
            await self.evaluation_metrics.measure_latency("process_articles", process_start_time)
            
            logger.info(f"Processed {len(valid_articles)} valid articles from {len(articles)} raw articles")
            return valid_articles
            
        except Exception as e:
            logger.error(f"Error processing articles: {e}")
            return []
            
    async def _generate_article_embeddings(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate embeddings for articles"""
        try:
            embedding_start_time = time.time()
            
            # Generate embeddings
            embeddings = await self.embedder.embed_articles(articles)
            
            # Measure latency
            await self.evaluation_metrics.measure_latency("generate_embeddings", embedding_start_time)
            
            logger.debug(f"Generated embeddings for {len(embeddings)} articles")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return {}
            
    async def _cluster_articles(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Cluster articles by topic"""
        try:
            cluster_start_time = time.time()
            
            # Cluster articles
            clustering_results = await self.topic_clustering.cluster_articles(articles)
            
            # Measure latency
            await self.evaluation_metrics.measure_latency("cluster_articles", cluster_start_time)
            
            clusters = clustering_results.get("clusters", {})
            logger.info(f"Created {len(clusters)} clusters")
            
            return clustering_results
            
        except Exception as e:
            logger.error(f"Error clustering articles: {e}")
            return {"clusters": {}, "topics": []}
            
    async def _get_rl_decision(self, clustering_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get RL agent decision on summarization"""
        try:
            clusters = clustering_results.get("clusters", {})
            
            # Prepare system metrics
            system_metrics = {
                "time_since_last_summary": self._get_time_since_last_summary(),
                "cpu_usage": 0.3,  # Placeholder - would come from system monitoring
                "memory_usage": 0.4,  # Placeholder
                "novelty_score": 0.7,  # Placeholder - would be calculated
                "latency": self.pipeline_metrics.get("avg_latency", 1.0)
            }
            
            # Get RL decision
            decision = await self.rl_agent.decide_summarization_action(clusters, system_metrics)
            
            logger.debug(f"RL agent decision: {decision['action']}")
            return decision
            
        except Exception as e:
            logger.error(f"Error getting RL decision: {e}")
            # Fallback decision
            return {
                "action": "summarize",
                "clusters": list(clustering_results.get("clusters", {}).keys()),
                "reason": "rl_fallback"
            }
            
    async def _execute_summarization(self, 
                                   clustering_results: Dict[str, Any],
                                   rl_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute summarization based on RL decision"""
        try:
            summarization_start_time = time.time()
            
            action = rl_decision.get("action", "wait")
            
            if action == "wait":
                logger.debug("RL agent decided to wait, skipping summarization")
                return {}
                
            clusters = clustering_results.get("clusters", {})
            target_clusters = rl_decision.get("clusters", list(clusters.keys()))
            
            # Filter clusters to process
            clusters_to_summarize = {
                cluster_id: clusters[cluster_id]
                for cluster_id in target_clusters
                if cluster_id in clusters
            }
            
            if not clusters_to_summarize:
                return {}
                
            # Generate summaries
            summaries = {}
            for cluster_id, cluster in clusters_to_summarize.items():
                try:
                    summary = await self.summarizer.summarize_cluster(cluster)
                    if summary and not summary.get("error"):
                        summaries[cluster_id] = summary
                except Exception as e:
                    logger.error(f"Error summarizing cluster {cluster_id}: {e}")
                    
            # Use incremental updater
            update_results = await self.incremental_updater.update_summary_stream(
                clusters_to_summarize, self.current_summaries
            )
            
            updated_summaries = update_results.get("summaries", {})
            self.current_summaries.update(updated_summaries)
            
            # Measure latency
            await self.evaluation_metrics.measure_latency("summarization", summarization_start_time)
            
            logger.info(f"Generated {len(summaries)} new summaries")
            return summaries
            
        except Exception as e:
            logger.error(f"Error executing summarization: {e}")
            return {}
            
    async def _update_memory_and_check_redundancy(self, summaries: Dict[str, Any]):
        """Update semantic memory and check for redundancy"""
        try:
            if not summaries:
                return
                
            # Generate embeddings for summaries
            summary_list = list(summaries.values())
            summary_embeddings = await self.embedder.embed_summaries(summary_list)
            
            # Check redundancy and update memory
            for cluster_id, summary in summaries.items():
                try:
                    embedding = summary_embeddings.get(cluster_id)
                    if embedding is None:
                        continue
                        
                    # Check if summary is redundant
                    redundancy_check = await self.semantic_memory.check_redundancy(
                        summary.get("summary", ""),
                        embedding,
                        summary.get("topic")
                    )
                    
                    summary["redundancy_check"] = redundancy_check
                    
                    # Store in memory if not redundant
                    if not redundancy_check.get("is_redundant", False):
                        memory_id = await self.semantic_memory.store_summary_memory(summary, embedding)
                        summary["memory_id"] = memory_id
                        logger.debug(f"Stored summary {cluster_id} in memory as {memory_id}")
                    else:
                        logger.debug(f"Summary {cluster_id} marked as redundant")
                        
                except Exception as e:
                    logger.error(f"Error processing memory for summary {cluster_id}: {e}")
                    
        except Exception as e:
            logger.error(f"Error updating memory: {e}")
            
    async def _index_content(self, 
                           articles: List[Dict[str, Any]],
                           summaries: Dict[str, Any],
                           article_embeddings: Dict[str, Any]):
        """Index articles and summaries"""
        try:
            index_start_time = time.time()
            
            # Index articles
            if articles:
                await self.search_index.index_articles(articles, article_embeddings)
                
            # Index summaries
            if summaries:
                summary_list = list(summaries.values())
                summary_embeddings = await self.embedder.embed_summaries(summary_list)
                await self.search_index.index_summaries(summary_list, summary_embeddings)
                
            # Measure indexing accuracy
            total_content = len(articles) + len(summaries)
            indexed_count = total_content  # Assume all indexed successfully for now
            accuracy = await self.evaluation_metrics.measure_index_accuracy(indexed_count, total_content)
            
            # Measure latency
            await self.evaluation_metrics.measure_latency("indexing", index_start_time)
            
            logger.debug(f"Indexed {len(articles)} articles and {len(summaries)} summaries")
            
        except Exception as e:
            logger.error(f"Error indexing content: {e}")
            
    async def _evaluate_cycle_performance(self, 
                                        cycle_start_time: float,
                                        summaries: Dict[str, Any]):
        """Evaluate performance of the processing cycle"""
        try:
            # Measure overall cycle latency
            cycle_latency = await self.evaluation_metrics.measure_latency("full_cycle", cycle_start_time)
            
            # Evaluate summary quality
            for cluster_id, summary in summaries.items():
                try:
                    # Get reference texts (articles in cluster)
                    reference_texts = []
                    articles = summary.get("articles", [])  # This would need to be passed from clustering
                    
                    cluster_info = {
                        "keywords": summary.get("keywords", []),
                        "topic": summary.get("topic", "")
                    }
                    
                    # Evaluate quality
                    quality_metrics = await self.evaluation_metrics.evaluate_summary_quality(
                        summary.get("summary", ""),
                        reference_texts,
                        cluster_info
                    )
                    
                    summary["quality_metrics"] = quality_metrics
                    
                    # Measure freshness
                    if "created_at" in summary:
                        freshness = await self.evaluation_metrics.measure_freshness(
                            summary["created_at"],
                            datetime.utcnow()
                        )
                        summary["freshness"] = freshness
                        
                except Exception as e:
                    logger.error(f"Error evaluating summary {cluster_id}: {e}")
                    
            # Update pipeline metrics
            self.pipeline_metrics["last_cycle_latency"] = cycle_latency
            self.pipeline_metrics["avg_latency"] = statistics.mean(
                self.evaluation_metrics.latency_measurements
            ) if self.evaluation_metrics.latency_measurements else 0.0
            
        except Exception as e:
            logger.error(f"Error evaluating cycle performance: {e}")
            
    async def _collect_and_process_feedback(self, 
                                          summaries: Dict[str, Any],
                                          rl_decision: Dict[str, Any]):
        """Collect feedback and update RL agent"""
        try:
            # Collect outcome metrics
            outcome_metrics = {
                "summaries_generated": len(summaries),
                "avg_quality": 0.0,
                "avg_latency": self.pipeline_metrics.get("last_cycle_latency", 0.0),
                "redundancy_rate": 0.0
            }
            
            if summaries:
                quality_scores = []
                redundant_count = 0
                
                for summary in summaries.values():
                    quality_metrics = summary.get("quality_metrics", {})
                    quality_score = quality_metrics.get("overall_quality", 0.0)
                    quality_scores.append(quality_score)
                    
                    redundancy_check = summary.get("redundancy_check", {})
                    if redundancy_check.get("is_redundant", False):
                        redundant_count += 1
                        
                outcome_metrics["avg_quality"] = statistics.mean(quality_scores)
                outcome_metrics["redundancy_rate"] = redundant_count / len(summaries)
                
            # Update RL agent
            await self.rl_agent.learn_from_feedback(rl_decision, outcome_metrics)
            
        except Exception as e:
            logger.error(f"Error collecting feedback: {e}")
            
    def _get_time_since_last_summary(self) -> float:
        """Get time since last summarization in seconds"""
        if not self.last_processing_time:
            return 0.0
            
        return (datetime.utcnow() - self.last_processing_time).total_seconds()
        
    async def search(self, 
                    query: str,
                    search_type: str = "hybrid",
                    filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for content using the search index"""
        try:
            if search_type == "text":
                return await self.search_index.search_text(query, filters=filters)
            elif search_type == "semantic":
                # Generate query embedding
                query_embedding = await self.embedder.embed_texts(query)
                if query_embedding.size > 0:
                    return await self.search_index.search_vector(query_embedding[0])
                return []
            else:  # hybrid
                query_embedding = await self.embedder.embed_texts(query)
                query_vector = query_embedding[0] if query_embedding.size > 0 else None
                return await self.search_index.hybrid_search(query, query_vector, filters=filters)
                
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []
            
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        try:
            # Component status
            component_status = {
                "gnews_client": self.gnews_client is not None,
                "text_processor": self.text_processor is not None,
                "topic_clustering": self.topic_clustering is not None,
                "summarizer": self.summarizer is not None,
                "embedder": self.embedder is not None,
                "search_index": self.search_index is not None,
                "semantic_memory": self.semantic_memory is not None,
                "rl_agent": self.rl_agent is not None,
                "evaluation_metrics": self.evaluation_metrics is not None
            }
            
            # Performance metrics
            performance_metrics = await self.evaluation_metrics.evaluate_real_time_performance()
            
            # Memory statistics
            memory_stats = await self.semantic_memory.get_memory_statistics()
            
            # RL agent statistics
            rl_stats = await self.rl_agent.get_agent_statistics()
            
            # Pipeline statistics
            pipeline_stats = {
                "running": self.running,
                "last_processing_time": self.last_processing_time,
                "current_summaries": len(self.current_summaries),
                "pipeline_metrics": self.pipeline_metrics
            }
            
            return {
                "timestamp": datetime.utcnow(),
                "component_status": component_status,
                "performance_metrics": performance_metrics,
                "memory_statistics": memory_stats,
                "rl_statistics": rl_stats,
                "pipeline_statistics": pipeline_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting pipeline status: {e}")
            return {"error": str(e)}
            
    async def stop(self):
        """Stop the pipeline gracefully"""
        logger.info("Stopping SummIndex pipeline...")
        self.running = False
        
    async def cleanup(self):
        """Cleanup all components"""
        try:
            logger.info("Cleaning up SummIndex pipeline...")
            
            # Cleanup components in reverse order
            cleanup_tasks = []
            
            if self.evaluation_metrics:
                cleanup_tasks.append(self.evaluation_metrics.cleanup())
                
            if self.rl_agent:
                cleanup_tasks.append(self.rl_agent.cleanup())
                
            if self.search_index:
                cleanup_tasks.append(self.search_index.cleanup())
                
            if self.semantic_memory:
                cleanup_tasks.append(self.semantic_memory.cleanup())
                
            if self.summarizer:
                cleanup_tasks.append(self.summarizer.cleanup())
                
            if self.incremental_updater:
                cleanup_tasks.append(self.incremental_updater.cleanup())
                
            if self.embedder:
                cleanup_tasks.append(self.embedder.cleanup())
                
            if self.topic_clustering:
                cleanup_tasks.append(self.topic_clustering.cleanup())
                
            if self.gnews_client:
                cleanup_tasks.append(self.gnews_client.cleanup())
                
            # Run cleanup tasks
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
            logger.info("SummIndex pipeline cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during pipeline cleanup: {e}")
