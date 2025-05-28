import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

logger = logging.getLogger(__name__)

# Request/Response models
class SearchRequest(BaseModel):
    query: str
    search_type: str = "hybrid"
    filters: Optional[Dict[str, Any]] = None
    size: int = 10

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_results: int
    search_type: str
    query: str
    timestamp: datetime

class SummaryRequest(BaseModel):
    cluster_id: Optional[str] = None
    topic: Optional[str] = None
    time_range_hours: int = 24

class FeedbackRequest(BaseModel):
    summary_id: str
    rating: float
    feedback_text: Optional[str] = None
    user_id: Optional[str] = None

class StatusResponse(BaseModel):
    status: str
    timestamp: datetime
    component_status: Dict[str, bool]
    performance_metrics: Dict[str, Any]
    message: str

class APIServer:
    """FastAPI server for SummIndex"""
    
    def __init__(self, pipeline, config):
        self.pipeline = pipeline
        self.config = config
        self.app = FastAPI(
            title="SummIndex API",
            description="Real-time news summarization and indexing system",
            version="1.0.0"
        )
        self._setup_routes()
        self._setup_middleware()
        
    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint"""
            return {
                "service": "SummIndex",
                "version": "1.0.0",
                "status": "running",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        @self.app.get("/health", response_model=StatusResponse)
        async def health_check():
            """Health check endpoint"""
            try:
                status = await self.pipeline.get_pipeline_status()
                
                # Determine overall health
                component_status = status.get("component_status", {})
                all_healthy = all(component_status.values())
                
                return StatusResponse(
                    status="healthy" if all_healthy else "degraded",
                    timestamp=datetime.utcnow(),
                    component_status=component_status,
                    performance_metrics=status.get("performance_metrics", {}),
                    message="All systems operational" if all_healthy else "Some components may be degraded"
                )
                
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return StatusResponse(
                    status="unhealthy",
                    timestamp=datetime.utcnow(),
                    component_status={},
                    performance_metrics={},
                    message=f"Health check failed: {str(e)}"
                )
                
        @self.app.post("/search", response_model=SearchResponse)
        async def search(request: SearchRequest):
            """Search for content"""
            try:
                logger.info(f"Search request: {request.query} ({request.search_type})")

                results = await self.pipeline.search(
                    query=request.query,
                    search_type=request.search_type,
                    filters=request.filters
                )

                limited_results = results[:request.size]
                
                return SearchResponse(
                    results=limited_results,
                    total_results=len(results),
                    search_type=request.search_type,
                    query=request.query,
                    timestamp=datetime.utcnow()
                )
                
            except Exception as e:
                logger.error(f"Search failed: {e}")
                raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
                
        @self.app.get("/summaries", response_model=Dict[str, Any])
        async def get_summaries(
            topic: Optional[str] = Query(None, description="Filter by topic"),
            hours: int = Query(24, description="Time range in hours"),
            limit: int = Query(50, description="Maximum number of summaries")
        ):
            """Get recent summaries"""
            try:
                # Get current summaries from pipeline
                current_summaries = self.pipeline.current_summaries
                
                # Apply filters
                filtered_summaries = []
                cutoff_time = datetime.utcnow().timestamp() - (hours * 3600)
                
                for summary in current_summaries.values():
                    # Filter by topic if specified
                    if topic and summary.get("topic") != topic:
                        continue
                        
                    # Filter by time
                    created_at = summary.get("created_at")
                    if created_at:
                        if isinstance(created_at, datetime):
                            created_timestamp = created_at.timestamp()
                        else:
                            try:
                                created_timestamp = datetime.fromisoformat(str(created_at)).timestamp()
                            except:
                                continue
                                
                        if created_timestamp < cutoff_time:
                            continue
                            
                    filtered_summaries.append(summary)
                    
                # Sort by creation time (newest first)
                filtered_summaries.sort(
                    key=lambda x: x.get("created_at", datetime.min),
                    reverse=True
                )
                
                # Limit results
                limited_summaries = filtered_summaries[:limit]
                
                return {
                    "summaries": limited_summaries,
                    "total_count": len(filtered_summaries),
                    "filters": {
                        "topic": topic,
                        "hours": hours,
                        "limit": limit
                    },
                    "timestamp": datetime.utcnow()
                }
                
            except Exception as e:
                logger.error(f"Get summaries failed: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get summaries: {str(e)}")
                
        @self.app.get("/summary/{cluster_id}", response_model=Dict[str, Any])
        async def get_summary(cluster_id: str):
            """Get specific summary by cluster ID"""
            try:
                summary = self.pipeline.current_summaries.get(cluster_id)
                
                if not summary:
                    raise HTTPException(status_code=404, detail="Summary not found")
                    
                # Get related context from semantic memory
                context_memories = []
                if self.pipeline.semantic_memory:
                    context_memories = await self.pipeline.semantic_memory.retrieve_context(
                        topic=summary.get("topic", ""),
                        max_memories=3
                    )
                    
                return {
                    "summary": summary,
                    "context": context_memories,
                    "timestamp": datetime.utcnow()
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Get summary failed: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get summary: {str(e)}")
                
        @self.app.post("/feedback", response_model=Dict[str, str])
        async def submit_feedback(request: FeedbackRequest):
            """Submit user feedback for a summary"""
            try:
                # Add feedback to evaluation metrics
                feedback_data = {
                    "rating": request.rating,
                    "feedback_text": request.feedback_text,
                    "user_id": request.user_id,
                    "timestamp": datetime.utcnow()
                }
                
                await self.pipeline.evaluation_metrics.add_human_feedback(
                    request.summary_id,
                    feedback_data
                )
                
                # Update RL agent with feedback
                if self.pipeline.rl_agent:
                    # Extract topic from summary if possible
                    summary = self.pipeline.current_summaries.get(request.summary_id)
                    topic = summary.get("topic", "general") if summary else "general"
                    
                    await self.pipeline.rl_agent.learn_from_feedback(
                        decision={"action": "feedback"},
                        outcome_metrics={"user_satisfaction": request.rating},
                        user_feedback={topic: request.rating}
                    )
                
                logger.info(f"Received feedback for summary {request.summary_id}: {request.rating}")
                
                return {
                    "status": "success",
                    "message": "Feedback submitted successfully",
                    "summary_id": request.summary_id
                }
                
            except Exception as e:
                logger.error(f"Submit feedback failed: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")
                
        @self.app.get("/topics", response_model=Dict[str, Any])
        async def get_topics():
            """Get available topics and their statistics"""
            try:
                topics_info = {}
                
                # Get topics from current summaries
                topic_counts = {}
                for summary in self.pipeline.current_summaries.values():
                    topic = summary.get("topic", "unknown")
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
                    
                # Get topic information from clustering
                if self.pipeline.topic_clustering:
                    # This would get more detailed topic information
                    pass
                    
                # Get similar topics from semantic memory
                similar_topics = {}
                if self.pipeline.semantic_memory:
                    for topic in topic_counts.keys():
                        try:
                            similar = await self.pipeline.semantic_memory.find_similar_topics(topic)
                            similar_topics[topic] = similar
                        except:
                            pass
                            
                return {
                    "topics": list(topic_counts.keys()),
                    "topic_counts": topic_counts,
                    "similar_topics": similar_topics,
                    "timestamp": datetime.utcnow()
                }
                
            except Exception as e:
                logger.error(f"Get topics failed: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get topics: {str(e)}")
                
        @self.app.get("/metrics", response_model=Dict[str, Any])
        async def get_metrics():
            try:
                # Get performance metrics
                performance_metrics = await self.pipeline.evaluation_metrics.evaluate_real_time_performance()
                
                # Get pipeline status
                pipeline_status = await self.pipeline.get_pipeline_status()
                
                # Get memory statistics
                memory_stats = {}
                if self.pipeline.semantic_memory:
                    memory_stats = await self.pipeline.semantic_memory.get_memory_statistics()
                    
                # Get RL agent statistics
                rl_stats = {}
                if self.pipeline.rl_agent:
                    rl_stats = await self.pipeline.rl_agent.get_agent_statistics()
                    
                # Get search index statistics
                index_stats = {}
                if self.pipeline.search_index:
                    index_stats = await self.pipeline.search_index.get_index_statistics()
                    
                return {
                    "performance_metrics": performance_metrics,
                    "pipeline_status": pipeline_status,
                    "memory_statistics": memory_stats,
                    "rl_statistics": rl_stats,
                    "index_statistics": index_stats,
                    "timestamp": datetime.utcnow()
                }
                
            except Exception as e:
                logger.error(f"Get metrics failed: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")
                
        @self.app.get("/evaluation/report", response_model=Dict[str, Any])
        async def get_evaluation_report():
            try:
                report = await self.pipeline.evaluation_metrics.generate_evaluation_report()
                return report
                
            except Exception as e:
                logger.error(f"Get evaluation report failed: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")
                
        @self.app.post("/admin/retrain", response_model=Dict[str, str])
        async def retrain_rl_agent(background_tasks: BackgroundTasks):
            try:
                if not self.pipeline.rl_agent:
                    raise HTTPException(status_code=404, detail="RL agent not available")
                    
                # Run training in background
                background_tasks.add_task(self._retrain_agent)
                
                return {
                    "status": "started",
                    "message": "RL agent retraining started in background"
                }
                
            except Exception as e:
                logger.error(f"Retrain failed: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to start retraining: {str(e)}")
                
        async def _retrain_agent(self):
            try:
                await self.pipeline.rl_agent.train_agent()
                logger.info("RL agent retraining completed")
            except Exception as e:
                logger.error(f"RL agent retraining failed: {e}")
                
        @self.app.exception_handler(Exception)
        async def global_exception_handler(request, exc):
            """Global exception handler"""
            logger.error(f"Unhandled exception: {exc}")
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"}
            )
            
    async def start(self):
        """Start the API server"""
        try:
            config = uvicorn.Config(
                self.app,
                host=self.config.API_HOST,
                port=self.config.API_PORT,
                log_level="info",
                access_log=True
            )
            
            server = uvicorn.Server(config)
            logger.info(f"Starting API server on {self.config.API_HOST}:{self.config.API_PORT}")
            
            await server.serve()
            
        except Exception as e:
            logger.error(f"Error starting API server: {e}")
            raise
            
    async def stop(self):
        """Stop the API server"""
        logger.info("Stopping API server...")
