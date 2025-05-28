import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np

# ROUGE metrics
from rouge_score import rouge_scorer

# BERTScore for semantic evaluation
try:
    from bert_score import score as bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("BERTScore not available, will skip semantic evaluation")

# Additional metrics
from scipy.stats import pearsonr
import statistics

logger = logging.getLogger(__name__)

class EvaluationMetrics:
    """Real-time evaluation metrics for summarization system"""
    
    def __init__(self, config: Any):
        self.config = config
        self.rouge_scorer = rouge_scorer.RougeScorer(
            self.config.ROUGE_TYPES, use_stemmer=True
        )
        
        # Metrics storage
        self.latency_measurements: deque = deque(maxlen=1000)
        self.quality_scores: deque = deque(maxlen=1000)
        self.freshness_scores: deque = deque(maxlen=1000)
        self.accuracy_scores: deque = deque(maxlen=1000)
        
        # Real-time tracking
        self.processing_times: Dict[str, List[float]] = defaultdict(list)
        self.summary_evaluations: List[Dict[str, Any]] = []
        self.system_performance: Dict[str, Any] = {}
        self.human_feedback: List[Dict[str, Any]] = []
        
        # Baseline metrics for comparison
        self.baseline_metrics: Dict[str, float] = {
            "rouge1": 0.3,
            "rouge2": 0.15,
            "rougeL": 0.25,
            "bert_score": 0.7,
            "latency": 2.0,
            "freshness": 0.8
        }
        
    async def initialize(self):
        """Initialize evaluation system"""
        logger.info("Initializing evaluation metrics system...")
        
    async def evaluate_summary_quality(self, 
                                     summary: str,
                                     reference_texts: List[str],
                                     cluster_info: Dict[str, Any]) -> Dict[str, float]:
        try:
            quality_metrics = {}
            
            # ROUGE scores
            rouge_scores = await self._calculate_rouge_scores(summary, reference_texts)
            quality_metrics.update(rouge_scores)
            
            # BERTScore (semantic similarity)
            if BERT_SCORE_AVAILABLE and reference_texts:
                bert_scores = await self._calculate_bert_score(summary, reference_texts)
                quality_metrics.update(bert_scores)
                
            # Length appropriateness
            length_score = self._evaluate_length_appropriateness(summary)
            quality_metrics["length_score"] = length_score
            
            # Keyword coverage
            keyword_coverage = self._evaluate_keyword_coverage(summary, cluster_info)
            quality_metrics["keyword_coverage"] = keyword_coverage
            
            # Coherence score (simple heuristic)
            coherence_score = self._evaluate_coherence(summary)
            quality_metrics["coherence"] = coherence_score
            
            # Novelty score
            novelty_score = self._evaluate_novelty(summary, reference_texts)
            quality_metrics["novelty"] = novelty_score
            
            # Overall quality score (weighted combination)
            overall_score = self._calculate_overall_quality(quality_metrics)
            quality_metrics["overall_quality"] = overall_score
            
            # Store for tracking
            self.quality_scores.append(overall_score)
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error evaluating summary quality: {e}")
            return {"overall_quality": 0.0}
            
    async def _calculate_rouge_scores(self, 
                                    summary: str,
                                    reference_texts: List[str]) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        try:
            if not reference_texts:
                return {rouge_type: 0.0 for rouge_type in self.config.ROUGE_TYPES}
                
            # Combine reference texts
            combined_reference = " ".join(reference_texts)
            
            # Calculate ROUGE scores
            rouge_scores = {}
            scores = self.rouge_scorer.score(combined_reference, summary)
            
            for rouge_type in self.config.ROUGE_TYPES:
                if rouge_type in scores:
                    rouge_scores[rouge_type] = scores[rouge_type].fmeasure
                else:
                    rouge_scores[rouge_type] = 0.0
                    
            return rouge_scores
            
        except Exception as e:
            logger.error(f"Error calculating ROUGE scores: {e}")
            return {rouge_type: 0.0 for rouge_type in self.config.ROUGE_TYPES}
            
    async def _calculate_bert_score(self, 
                                  summary: str,
                                  reference_texts: List[str]) -> Dict[str, float]:
        """Calculate BERTScore for semantic similarity"""
        try:
            if not BERT_SCORE_AVAILABLE or not reference_texts:
                return {"bert_score": 0.0}
                
            # Use asyncio to run BERTScore in executor
            def compute_bert_score():
                P, R, F1 = bert_score([summary], reference_texts, lang="en")
                return {
                    "bert_precision": float(P.mean()),
                    "bert_recall": float(R.mean()),
                    "bert_score": float(F1.mean())
                }
                
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, compute_bert_score)
            
        except Exception as e:
            logger.error(f"Error calculating BERTScore: {e}")
            return {"bert_score": 0.0}
            
    def _evaluate_length_appropriateness(self, summary: str) -> float:
        """Evaluate if summary length is appropriate"""
        word_count = len(summary.split())
        
        min_length = self.config.MIN_SUMMARY_LENGTH
        max_length = self.config.MAX_SUMMARY_LENGTH
        target_length = (min_length + max_length) / 2
        
        if min_length <= word_count <= max_length:
            # Score based on how close to target length
            distance_from_target = abs(word_count - target_length)
            max_distance = (max_length - min_length) / 2
            score = 1.0 - (distance_from_target / max_distance)
            return max(0.0, score)
        else:
            # Penalty for being outside range
            if word_count < min_length:
                return max(0.0, word_count / min_length)
            else:
                return max(0.0, max_length / word_count)
                
    def _evaluate_keyword_coverage(self, 
                                 summary: str,
                                 cluster_info: Dict[str, Any]) -> float:
        """Evaluate how well summary covers important keywords"""
        try:
            keywords = cluster_info.get("keywords", [])
            if not keywords:
                return 0.5  # Neutral score if no keywords
                
            summary_lower = summary.lower()
            covered_keywords = sum(1 for kw in keywords if kw.lower() in summary_lower)
            
            return covered_keywords / len(keywords)
            
        except Exception as e:
            logger.error(f"Error evaluating keyword coverage: {e}")
            return 0.0
            
    def _evaluate_coherence(self, summary: str) -> float:
        """Evaluate summary coherence using simple heuristics"""
        try:
            sentences = self._split_sentences(summary)
            if len(sentences) <= 1:
                return 0.5
                
            # Check for proper sentence structure
            coherence_score = 0.0
            
            # Check sentence lengths (not too short or too long)
            sentence_lengths = [len(s.split()) for s in sentences]
            avg_length = np.mean(sentence_lengths)
            if 5 <= avg_length <= 25:
                coherence_score += 0.3
                
            # Check for proper punctuation
            proper_endings = sum(1 for s in sentences if s.strip().endswith(('.', '!', '?')))
            if proper_endings / len(sentences) >= 0.8:
                coherence_score += 0.3
                
            # Check for transition words/phrases (simple heuristic)
            transition_words = ["however", "therefore", "moreover", "furthermore", "additionally"]
            has_transitions = any(word in summary.lower() for word in transition_words)
            if has_transitions:
                coherence_score += 0.2
                
            # Check for repetition (negative indicator)
            words = summary.lower().split()
            if len(set(words)) / len(words) >= 0.7:  # Low repetition
                coherence_score += 0.2
                
            return min(1.0, coherence_score)
            
        except Exception as e:
            logger.error(f"Error evaluating coherence: {e}")
            return 0.5
            
    def _evaluate_novelty(self, summary: str, reference_texts: List[str]) -> float:
        """Evaluate novelty of summary compared to original texts"""
        try:
            if not reference_texts:
                return 0.5
                
            summary_words = set(summary.lower().split())
            reference_words = set()
            
            for text in reference_texts:
                reference_words.update(text.lower().split())
                
            if not reference_words:
                return 0.5
                
            # Calculate how many summary words are not in references
            novel_words = summary_words - reference_words
            novelty_ratio = len(novel_words) / len(summary_words) if summary_words else 0
            
            # Optimal novelty is moderate (around 0.3-0.5)
            if 0.2 <= novelty_ratio <= 0.6:
                return 1.0 - abs(novelty_ratio - 0.4) / 0.4
            else:
                return max(0.0, 1.0 - abs(novelty_ratio - 0.4) / 0.6)
                
        except Exception as e:
            logger.error(f"Error evaluating novelty: {e}")
            return 0.5
            
    def _calculate_overall_quality(self, metrics: Dict[str, float]) -> float:
        """Calculate overall quality score from individual metrics"""
        weights = {
            "rouge1": 0.25,
            "rouge2": 0.15,
            "rougeL": 0.15,
            "bert_score": 0.20,
            "length_score": 0.10,
            "keyword_coverage": 0.15,
            "coherence": 0.10,
            "novelty": 0.05
        }
        
        overall_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                overall_score += metrics[metric] * weight
                total_weight += weight
                
        return overall_score / total_weight if total_weight > 0 else 0.0
        
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
        
    async def measure_latency(self, operation: str, start_time: float) -> float:
        """Measure and record latency for operations"""
        latency = time.time() - start_time
        
        self.processing_times[operation].append(latency)
        self.latency_measurements.append(latency)
        
        # Check against target
        if latency > self.config.TARGET_LATENCY:
            logger.warning(f"Latency target exceeded: {latency:.2f}s > {self.config.TARGET_LATENCY}s")
            
        return latency
        
    async def measure_freshness(self, 
                              content_timestamp: datetime,
                              processing_timestamp: datetime) -> float:
        """Measure content freshness"""
        try:
            time_diff = (processing_timestamp - content_timestamp).total_seconds()
            
            # Freshness decreases with time
            # 100% fresh if processed within 1 minute
            # 50% fresh if processed within 1 hour
            # Approaches 0% as time increases
            
            if time_diff <= 60:  # 1 minute
                freshness = 1.0
            elif time_diff <= 3600:  # 1 hour
                freshness = 1.0 - (time_diff - 60) / (3600 - 60) * 0.5
            else:
                freshness = max(0.1, 0.5 * np.exp(-(time_diff - 3600) / 7200))
                
            self.freshness_scores.append(freshness)
            return freshness
            
        except Exception as e:
            logger.error(f"Error measuring freshness: {e}")
            return 0.5
            
    async def measure_index_accuracy(self, 
                                   indexed_count: int,
                                   total_count: int) -> float:
        """Measure indexing accuracy"""
        if total_count == 0:
            return 1.0
            
        accuracy = indexed_count / total_count
        self.accuracy_scores.append(accuracy)
        
        if accuracy < self.config.TARGET_ACCURACY:
            logger.warning(f"Index accuracy below target: {accuracy:.3f} < {self.config.TARGET_ACCURACY}")
            
        return accuracy
        
    async def evaluate_real_time_performance(self) -> Dict[str, float]:
        """Evaluate overall real-time performance"""
        try:
            metrics = {}
            
            # Latency metrics
            if self.latency_measurements:
                metrics["avg_latency"] = statistics.mean(self.latency_measurements)
                metrics["p95_latency"] = np.percentile(self.latency_measurements, 95)
                metrics["p99_latency"] = np.percentile(self.latency_measurements, 99)
                metrics["latency_target_met"] = float(
                    metrics["avg_latency"] <= self.config.TARGET_LATENCY
                )
            else:
                metrics.update({
                    "avg_latency": 0.0,
                    "p95_latency": 0.0,
                    "p99_latency": 0.0,
                    "latency_target_met": 0.0
                })
                
            # Quality metrics
            if self.quality_scores:
                metrics["avg_quality"] = statistics.mean(self.quality_scores)
                metrics["quality_trend"] = self._calculate_trend(self.quality_scores)
            else:
                metrics.update({
                    "avg_quality": 0.0,
                    "quality_trend": 0.0
                })
                
            # Freshness metrics
            if self.freshness_scores:
                metrics["avg_freshness"] = statistics.mean(self.freshness_scores)
                metrics["freshness_trend"] = self._calculate_trend(self.freshness_scores)
            else:
                metrics.update({
                    "avg_freshness": 0.0,
                    "freshness_trend": 0.0
                })
                
            # Accuracy metrics
            if self.accuracy_scores:
                metrics["avg_accuracy"] = statistics.mean(self.accuracy_scores)
                metrics["accuracy_target_met"] = float(
                    metrics["avg_accuracy"] >= self.config.TARGET_ACCURACY
                )
            else:
                metrics.update({
                    "avg_accuracy": 0.0,
                    "accuracy_target_met": 0.0
                })
                
            # Update frequency
            recent_evaluations = [
                e for e in self.summary_evaluations
                if (datetime.utcnow() - e["timestamp"]).total_seconds() < 3600
            ]
            metrics["evaluations_per_hour"] = len(recent_evaluations)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating real-time performance: {e}")
            return {}
            
    def _calculate_trend(self, values: deque, window_size: int = 10) -> float:
        """Calculate trend in recent values (-1 to 1, negative = declining)"""
        if len(values) < window_size:
            return 0.0
            
        recent_values = list(values)[-window_size:]
        x = np.arange(len(recent_values))
        
        try:
            correlation, _ = pearsonr(x, recent_values)
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0
            
    async def add_human_feedback(self, 
                               summary_id: str,
                               feedback: Dict[str, Any]):
        """Add human evaluation feedback"""
        feedback_entry = {
            "summary_id": summary_id,
            "timestamp": datetime.utcnow(),
            **feedback
        }
        
        self.human_feedback.append(feedback_entry)
        
        # Limit storage
        if len(self.human_feedback) > 1000:
            self.human_feedback = self.human_feedback[-500:]
            
        logger.info(f"Added human feedback for summary {summary_id}")
        
    async def generate_evaluation_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        try:
            performance_metrics = await self.evaluate_real_time_performance()
            
            # Recent human feedback analysis
            recent_feedback = [
                f for f in self.human_feedback
                if (datetime.utcnow() - f["timestamp"]).total_seconds() < 86400  # 24 hours
            ]
            
            human_metrics = {}
            if recent_feedback:
                ratings = [f.get("rating", 0) for f in recent_feedback if "rating" in f]
                if ratings:
                    human_metrics["avg_human_rating"] = statistics.mean(ratings)
                    human_metrics["human_feedback_count"] = len(ratings)
                    
            # Processing time breakdown
            processing_breakdown = {}
            for operation, times in self.processing_times.items():
                if times:
                    processing_breakdown[operation] = {
                        "avg_time": statistics.mean(times),
                        "count": len(times)
                    }
                    
            # System targets
            targets_met = {
                "latency_target": performance_metrics.get("latency_target_met", 0.0),
                "accuracy_target": performance_metrics.get("accuracy_target_met", 0.0),
                "overall_performance": (
                    performance_metrics.get("latency_target_met", 0.0) +
                    performance_metrics.get("accuracy_target_met", 0.0)
                ) / 2.0
            }
            
            report = {
                "timestamp": datetime.utcnow(),
                "performance_metrics": performance_metrics,
                "human_feedback": human_metrics,
                "processing_breakdown": processing_breakdown,
                "targets_met": targets_met,
                "baseline_comparison": self._compare_to_baseline(performance_metrics),
                "recommendations": self._generate_recommendations(performance_metrics)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating evaluation report: {e}")
            return {"error": str(e)}
            
    def _compare_to_baseline(self, current_metrics: Dict[str, float]) -> Dict[str, float]:
        """Compare current metrics to baseline"""
        comparison = {}
        
        for metric, baseline_value in self.baseline_metrics.items():
            current_value = current_metrics.get(metric)
            if current_value is not None:
                improvement = (current_value - baseline_value) / baseline_value
                comparison[f"{metric}_improvement"] = improvement
                
        return comparison
        
    def _generate_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Generate improvement recommendations based on metrics"""
        recommendations = []
        
        # Latency recommendations
        avg_latency = metrics.get("avg_latency", 0)
        if avg_latency > self.config.TARGET_LATENCY:
            recommendations.append(
                f"Consider optimizing model inference or using faster models. "
                f"Current latency: {avg_latency:.2f}s, target: {self.config.TARGET_LATENCY}s"
            )
            
        # Quality recommendations
        avg_quality = metrics.get("avg_quality", 0)
        if avg_quality < 0.7:
            recommendations.append(
                f"Summary quality could be improved. Current: {avg_quality:.2f}, "
                "consider fine-tuning models or adjusting parameters."
            )
            
        # Accuracy recommendations
        avg_accuracy = metrics.get("avg_accuracy", 0)
        if avg_accuracy < self.config.TARGET_ACCURACY:
            recommendations.append(
                f"Indexing accuracy below target. Current: {avg_accuracy:.3f}, "
                f"target: {self.config.TARGET_ACCURACY}. Check indexing pipeline."
            )
            
        # Freshness recommendations
        avg_freshness = metrics.get("avg_freshness", 0)
        if avg_freshness < 0.8:
            recommendations.append(
                f"Content freshness could be improved. Current: {avg_freshness:.2f}, "
                "consider reducing processing delays."
            )
            
        if not recommendations:
            recommendations.append("System performance is meeting all targets.")
            
        return recommendations
        
    async def cleanup(self):
        """Cleanup evaluation resources"""
        # Save evaluation data for analysis
        try:
            evaluation_data = {
                "summary_evaluations": self.summary_evaluations,
                "human_feedback": self.human_feedback,
                "processing_times": dict(self.processing_times)
            }
            
            # In a production system, you might save this to a database or file
            logger.info("Evaluation data prepared for archival")
            
        except Exception as e:
            logger.error(f"Error during evaluation cleanup: {e}")
            
        # Clear memory
        self.latency_measurements.clear()
        self.quality_scores.clear()
        self.freshness_scores.clear()
        self.accuracy_scores.clear()
        self.summary_evaluations.clear()
        self.human_feedback.clear()
        self.processing_times.clear()
        
        logger.info("Evaluation metrics cleanup completed")
