import logging
import asyncio
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)

class IncrementalUpdater:

    def __init__(self, config: Any):
        self.config = config
        self.summary_cache: Dict[str, Dict[str, Any]] = {}
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.update_queue: List[Dict[str, Any]] = []
        self.last_update_time: Dict[str, datetime] = {}
        self.change_detection: Dict[str, str] = {}  # cluster_id -> content_hash
        
    async def initialize(self):
        logger.info("Initializing incremental updater...")
        
    async def update_summary_stream(self, 
                                  new_clusters: Dict[str, Dict[str, Any]],
                                  existing_summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:

        try:
            start_time = datetime.utcnow()
            
            # Detect changes in clusters
            changes = await self._detect_cluster_changes(new_clusters)
            
            # Determine what needs to be updated
            update_plan = await self._create_update_plan(changes, existing_summaries)
            
            # Execute incremental updates
            updated_summaries = await self._execute_updates(
                update_plan, new_clusters, existing_summaries
            )
            
            # Update metadata
            await self._update_metadata(updated_summaries)
            
            update_time = datetime.utcnow()
            
            return {
                "summaries": updated_summaries,
                "update_info": {
                    "total_clusters": len(new_clusters),
                    "updated_summaries": len(updated_summaries),
                    "new_summaries": len(changes.get("new", [])),
                    "modified_summaries": len(changes.get("modified", [])),
                    "removed_summaries": len(changes.get("removed", [])),
                    "processing_time": (update_time - start_time).total_seconds(),
                    "timestamp": update_time
                }
            }
            
        except Exception as e:
            logger.error(f"Error updating summary stream: {e}")
            return {
                "summaries": existing_summaries,
                "update_info": {"error": str(e)}
            }
            
    async def _detect_cluster_changes(self, 
                                    new_clusters: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:

        changes = {
            "new": [],
            "modified": [],
            "removed": [],
            "unchanged": []
        }
        
        current_cluster_ids = set(new_clusters.keys())
        previous_cluster_ids = set(self.change_detection.keys())
        
        # Find new clusters
        new_cluster_ids = current_cluster_ids - previous_cluster_ids
        changes["new"] = list(new_cluster_ids)
        
        # Find removed clusters
        removed_cluster_ids = previous_cluster_ids - current_cluster_ids
        changes["removed"] = list(removed_cluster_ids)
        
        # Check for modifications in existing clusters
        for cluster_id in current_cluster_ids & previous_cluster_ids:
            current_hash = self._calculate_cluster_hash(new_clusters[cluster_id])
            previous_hash = self.change_detection.get(cluster_id, "")
            
            if current_hash != previous_hash:
                changes["modified"].append(cluster_id)
            else:
                changes["unchanged"].append(cluster_id)
                
        # Update change detection hashes
        for cluster_id, cluster in new_clusters.items():
            self.change_detection[cluster_id] = self._calculate_cluster_hash(cluster)
            
        # Clean up removed clusters
        for cluster_id in removed_cluster_ids:
            if cluster_id in self.change_detection:
                del self.change_detection[cluster_id]
                
        logger.debug(f"Cluster changes: {len(changes['new'])} new, "
                    f"{len(changes['modified'])} modified, "
                    f"{len(changes['removed'])} removed")
                    
        return changes
        
    def _calculate_cluster_hash(self, cluster: Dict[str, Any]) -> str:
        """Calculate hash for cluster to detect changes"""
        # Create a stable representation of cluster for hashing
        articles = cluster.get("articles", [])
        article_ids = sorted([article.get("id", "") for article in articles])
        
        # Include cluster metadata
        hash_content = {
            "article_ids": article_ids,
            "keywords": sorted(cluster.get("keywords", [])),
            "topic": cluster.get("topic", ""),
            "size": cluster.get("size", 0)
        }
        
        hash_string = str(hash_content)
        return hashlib.md5(hash_string.encode()).hexdigest()
        
    async def _create_update_plan(self, 
                                changes: Dict[str, List[str]],
                                existing_summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        update_plan = {
            "immediate": [],  # High priority updates
            "batch": [],      # Can be batched together
            "cleanup": [],    # Cleanup operations
            "dependencies": {}
        }
        
        # New clusters need immediate summarization
        for cluster_id in changes["new"]:
            update_plan["immediate"].append({
                "cluster_id": cluster_id,
                "operation": "create",
                "priority": "high"
            })
            
        # Modified clusters need re-summarization
        for cluster_id in changes["modified"]:
            # Check if we can do incremental update or need full re-summarization
            existing_summary = existing_summaries.get(cluster_id)
            
            if existing_summary and self._can_incremental_update(cluster_id, existing_summary):
                update_plan["batch"].append({
                    "cluster_id": cluster_id,
                    "operation": "incremental_update",
                    "priority": "medium"
                })
            else:
                update_plan["immediate"].append({
                    "cluster_id": cluster_id,
                    "operation": "full_update",
                    "priority": "high"
                })
                
        # Removed clusters need cleanup
        for cluster_id in changes["removed"]:
            update_plan["cleanup"].append({
                "cluster_id": cluster_id,
                "operation": "remove",
                "priority": "low"
            })
            
        return update_plan
        
    def _can_incremental_update(self, 
                              cluster_id: str, 
                              existing_summary: Dict[str, Any]) -> bool:
        # Check if summary is recent enough
        created_at = existing_summary.get("created_at")
        if created_at:
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                except:
                    return False
                    
            age = datetime.utcnow() - created_at
            if age > timedelta(hours=6):  # Too old for incremental update
                return False
                
        # Check if the change is small enough for incremental update
        # This is a simplified check - in practice, you'd analyze the specific changes
        return True
        
    async def _execute_updates(self,
                             update_plan: Dict[str, Any],
                             new_clusters: Dict[str, Dict[str, Any]],
                             existing_summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:

        updated_summaries = existing_summaries.copy()
        
        # Execute immediate updates first
        for update_item in update_plan["immediate"]:
            cluster_id = update_item["cluster_id"]
            operation = update_item["operation"]
            
            if operation in ["create", "full_update"] and cluster_id in new_clusters:
                # This would normally call the summarizer
                # For now, we'll create a placeholder summary
                summary = await self._create_incremental_summary(
                    new_clusters[cluster_id], operation
                )
                updated_summaries[cluster_id] = summary
                
        # Execute batch updates
        batch_updates = []
        for update_item in update_plan["batch"]:
            cluster_id = update_item["cluster_id"]
            if cluster_id in new_clusters:
                batch_updates.append((cluster_id, new_clusters[cluster_id]))
                
        if batch_updates:
            batch_summaries = await self._execute_batch_updates(batch_updates)
            updated_summaries.update(batch_summaries)
            
        # Execute cleanup
        for update_item in update_plan["cleanup"]:
            cluster_id = update_item["cluster_id"]
            if cluster_id in updated_summaries:
                del updated_summaries[cluster_id]
                
        return updated_summaries
        
    async def _create_incremental_summary(self, 
                                        cluster: Dict[str, Any],
                                        operation: str) -> Dict[str, Any]:
        # This is a simplified implementation
        # In the full system, this would integrate with the MultiModelSummarizer
        
        articles = cluster.get("articles", [])
        
        # Create basic summary information
        summary = {
            "cluster_id": cluster.get("id"),
            "topic": cluster.get("topic", "unknown"),
            "summary": f"Summary of {len(articles)} articles about {cluster.get('topic', 'general news')}",
            "model_used": "incremental",
            "article_count": len(articles),
            "keywords": cluster.get("keywords", []),
            "sources": cluster.get("sources", []),
            "created_at": datetime.utcnow(),
            "operation": operation,
            "word_count": 20,  # Placeholder
            "quality_score": 0.7  # Placeholder
        }
        
        return summary
        
    async def _execute_batch_updates(self, 
                                   batch_items: List[tuple]) -> Dict[str, Dict[str, Any]]:
        batch_summaries = {}
        
        # Process batch items - this could be parallelized
        for cluster_id, cluster_data in batch_items:
            summary = await self._create_incremental_summary(cluster_data, "incremental_update")
            batch_summaries[cluster_id] = summary
            
        return batch_summaries
        
    async def _update_metadata(self, summaries: Dict[str, Dict[str, Any]]):
        """Update metadata and caches"""
        current_time = datetime.utcnow()
        
        for cluster_id, summary in summaries.items():
            self.last_update_time[cluster_id] = current_time
            
        # Update summary cache
        self.summary_cache.update(summaries)
        
        # Cleanup old entries
        await self._cleanup_old_metadata()
        
    async def _cleanup_old_metadata(self):
        """Clean up old metadata to prevent memory leaks"""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        # Clean up old update times
        expired_clusters = [
            cluster_id for cluster_id, update_time in self.last_update_time.items()
            if update_time < cutoff_time
        ]
        
        for cluster_id in expired_clusters:
            if cluster_id in self.last_update_time:
                del self.last_update_time[cluster_id]
            if cluster_id in self.summary_cache:
                del self.summary_cache[cluster_id]
                
        if expired_clusters:
            logger.info(f"Cleaned up metadata for {len(expired_clusters)} expired clusters")
            
    async def get_update_statistics(self) -> Dict[str, Any]:
        """Get statistics about incremental updates"""
        current_time = datetime.utcnow()
        
        # Calculate update frequencies
        recent_updates = sum(
            1 for update_time in self.last_update_time.values()
            if (current_time - update_time).total_seconds() < 3600  # Last hour
        )
        
        return {
            "total_tracked_clusters": len(self.last_update_time),
            "cached_summaries": len(self.summary_cache),
            "recent_updates": recent_updates,
            "queue_size": len(self.update_queue),
            "average_update_interval": self._calculate_average_update_interval(),
            "last_cleanup": current_time
        }
        
    def _calculate_average_update_interval(self) -> float:
        """Calculate average time between updates"""
        if len(self.last_update_time) < 2:
            return 0.0
            
        update_times = sorted(self.last_update_time.values())
        intervals = [
            (update_times[i+1] - update_times[i]).total_seconds()
            for i in range(len(update_times) - 1)
        ]
        
        return sum(intervals) / len(intervals) if intervals else 0.0
        
    async def force_full_update(self, cluster_ids: Optional[List[str]] = None):
        """Force full update for specified clusters or all clusters"""
        if cluster_ids is None:
            cluster_ids = list(self.change_detection.keys())
            
        # Clear change detection for forced updates
        for cluster_id in cluster_ids:
            if cluster_id in self.change_detection:
                self.change_detection[cluster_id] = ""
                
        logger.info(f"Forced full update for {len(cluster_ids)} clusters")
        
    async def cleanup(self):
        """Cleanup incremental updater resources"""
        self.summary_cache.clear()
        self.dependency_graph.clear()
        self.update_queue.clear()
        self.last_update_time.clear()
        self.change_detection.clear()
        
        logger.info("Incremental updater cleaned up")
