import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import torch
from transformers import (
    PegasusForConditionalGeneration, PegasusTokenizer,
    BartForConditionalGeneration, BartTokenizer,
    T5ForConditionalGeneration, T5Tokenizer,
    LongformerTokenizer, LEDForConditionalGeneration
)
import numpy as np
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class MultiModelSummarizer:    
    def __init__(self, config: Any):
        self.config = config
        self.models: Dict[str, Dict[str, Any]] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.model_performance: Dict[str, Dict[str, float]] = {}
        
    async def initialize(self):
        try:
            logger.info("Initializing summarization models...")
            
            # Initialize models based on configuration
            await self._load_pegasus_model()
            await self._load_bart_model()
            await self._load_t5_model()
            
            # Initialize performance tracking
            for model_name in self.models.keys():
                self.model_performance[model_name] = {
                    "avg_latency": 0.0,
                    "success_rate": 1.0,
                    "total_requests": 0,
                    "failed_requests": 0
                }
                
            logger.info(f"Initialized {len(self.models)} summarization models")
            
        except Exception as e:
            logger.error(f"Error initializing summarization models: {e}")
            raise
            
    async def _load_pegasus_model(self):
        try:
            model_name = self.config.SUMMARIZATION_MODELS["pegasus"]
            
            def load_model():
                tokenizer = PegasusTokenizer.from_pretrained(model_name)
                model = PegasusForConditionalGeneration.from_pretrained(model_name)
                model.to(self.device)
                model.eval()
                return tokenizer, model
                
            loop = asyncio.get_event_loop()
            tokenizer, model = await loop.run_in_executor(self.executor, load_model)
            
            self.models["pegasus"] = {
                "tokenizer": tokenizer,
                "model": model,
                "max_length": 1024,
                "type": "abstractive"
            }
            
            logger.info("PEGASUS model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading PEGASUS model: {e}")
            
    async def _load_bart_model(self):
        """Load BART model for balanced speed and quality"""
        try:
            model_name = self.config.SUMMARIZATION_MODELS["bart"]
            
            def load_model():
                tokenizer = BartTokenizer.from_pretrained(model_name)
                model = BartForConditionalGeneration.from_pretrained(model_name)
                model.to(self.device)
                model.eval()
                return tokenizer, model
                
            loop = asyncio.get_event_loop()
            tokenizer, model = await loop.run_in_executor(self.executor, load_model)
            
            self.models["bart"] = {
                "tokenizer": tokenizer,
                "model": model,
                "max_length": 1024,
                "type": "abstractive"
            }
            
            logger.info("BART model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading BART model: {e}")
            
    async def _load_t5_model(self):
        try:
            model_name = self.config.SUMMARIZATION_MODELS["t5"]
            
            def load_model():
                tokenizer = T5Tokenizer.from_pretrained(model_name)
                model = T5ForConditionalGeneration.from_pretrained(model_name)
                model.to(self.device)
                model.eval()
                return tokenizer, model
                
            loop = asyncio.get_event_loop()
            tokenizer, model = await loop.run_in_executor(self.executor, load_model)
            
            self.models["t5"] = {
                "tokenizer": tokenizer,
                "model": model,
                "max_length": 512,
                "type": "abstractive"
            }
            
            logger.info("T5 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading T5 model: {e}")
            
    async def summarize_cluster(self, 
                              cluster: Dict[str, Any],
                              model_preference: Optional[str] = None) -> Dict[str, Any]:
        try:
            start_time = time.time()
            
            articles = cluster.get("articles", [])
            if not articles:
                return self._create_empty_summary(cluster)
                
            # Combine article texts
            combined_text = self._combine_article_texts(articles)
            
            # Choose the best model for this task
            model_name = self._select_model(combined_text, model_preference)
            
            if not model_name or model_name not in self.models:
                logger.warning("No suitable model available, using extractive fallback")
                return await self._extractive_summarize(cluster, combined_text)
                
            # Generate summary using selected model
            summary_text = await self._generate_summary(combined_text, model_name)
            
            # Post-process summary
            summary_info = self._create_summary_info(
                cluster, summary_text, model_name, start_time
            )
            
            # Update model performance
            self._update_model_performance(model_name, start_time, success=True)
            
            return summary_info
            
        except Exception as e:
            logger.error(f"Error summarizing cluster: {e}")
            if model_preference:
                self._update_model_performance(model_preference, start_time, success=False)
            return await self._extractive_summarize(cluster, combined_text)
            
    def _combine_article_texts(self, articles: List[Dict[str, Any]]) -> str:
        """Combine article texts for summarization"""
        texts = []
        
        for article in articles:
            # Use clean text if available
            text = article.get("full_text_clean") or article.get("full_text", "")
            if text:
                # Add source attribution
                source = article.get("source", "Unknown")
                texts.append(f"[{source}] {text}")
                
        return " ".join(texts)
        
    def _select_model(self, text: str, preference: Optional[str] = None) -> Optional[str]:
        if preference and preference in self.models:
            return preference
            
        text_length = len(text.split())
        
        # Model selection logic based on text length and performance
        if text_length > 800:
            # For long texts, prefer models that handle longer sequences
            if "longformer" in self.models:
                return "longformer"
            elif "bart" in self.models:
                return "bart"
                
        # For medium length texts, prefer high-quality models
        elif text_length > 200:
            if "pegasus" in self.models and self._is_model_performing_well("pegasus"):
                return "pegasus"
            elif "bart" in self.models and self._is_model_performing_well("bart"):
                return "bart"
                
        # For shorter texts, use faster models
        else:
            if "t5" in self.models and self._is_model_performing_well("t5"):
                return "t5"
            elif "bart" in self.models:
                return "bart"
                
        # Fallback to any available model
        available_models = list(self.models.keys())
        return available_models[0] if available_models else None
        
    def _is_model_performing_well(self, model_name: str) -> bool:
        perf = self.model_performance.get(model_name, {})
        
        # Consider a model performing well if:
        # - Success rate > 90%
        # - Average latency < 5 seconds
        success_rate = perf.get("success_rate", 0.0)
        avg_latency = perf.get("avg_latency", float('inf'))
        
        return success_rate > 0.9 and avg_latency < 5.0
        
    async def _generate_summary(self, text: str, model_name: str) -> str:
        model_info = self.models[model_name]
        tokenizer = model_info["tokenizer"]
        model = model_info["model"]
        max_length = model_info["max_length"]
        
        def generate():
            try:
                # Prepare input based on model type
                if model_name == "t5":
                    input_text = f"summarize: {text}"
                else:
                    input_text = text
                    
                # Tokenize input
                inputs = tokenizer(
                    input_text,
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                    padding=True
                )
                
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate summary
                with torch.no_grad():
                    summary_ids = model.generate(
                        **inputs,
                        max_length=self.config.MAX_SUMMARY_LENGTH,
                        min_length=self.config.MIN_SUMMARY_LENGTH,
                        num_beams=4,
                        length_penalty=2.0,
                        early_stopping=True,
                        no_repeat_ngram_size=3
                    )
                    
                # Decode summary
                summary = tokenizer.decode(
                    summary_ids[0], 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                
                return summary
                
            except Exception as e:
                logger.error(f"Error in model generation: {e}")
                raise
                
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, generate)
        
    async def _extractive_summarize(self, 
                                  cluster: Dict[str, Any], 
                                  text: str) -> Dict[str, Any]:
        try:
            # Simple extractive summarization
            sentences = self._split_sentences(text)
            
            if len(sentences) <= 3:
                summary_text = " ".join(sentences)
            else:
                # Score sentences based on word frequency and position
                scored_sentences = self._score_sentences(sentences)
                
                # Select top sentences
                num_sentences = min(3, len(scored_sentences))
                top_sentences = sorted(scored_sentences[:num_sentences], 
                                     key=lambda x: sentences.index(x[1]))
                
                summary_text = " ".join([sent[1] for sent in top_sentences])
                
            return self._create_summary_info(
                cluster, summary_text, "extractive", time.time()
            )
            
        except Exception as e:
            logger.error(f"Error in extractive summarization: {e}")
            return self._create_empty_summary(cluster)
            
    def _split_sentences(self, text: str) -> List[str]:
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.split()) > 3]
        
    def _score_sentences(self, sentences: List[str]) -> List[tuple]:
        from collections import Counter
        import re
        
        # Get word frequencies
        all_words = []
        for sentence in sentences:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', sentence.lower())
            all_words.extend(words)
            
        word_freq = Counter(all_words)
        
        # Score each sentence
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            words = re.findall(r'\b[a-zA-Z]{3,}\b', sentence.lower())
            score = sum(word_freq[word] for word in words)
            
            # Boost score for earlier sentences
            position_boost = 1.0 - (i / len(sentences)) * 0.3
            score *= position_boost
            
            sentence_scores.append((score, sentence))
            
        return sorted(sentence_scores, key=lambda x: x[0], reverse=True)
        
    def _create_summary_info(self, 
                           cluster: Dict[str, Any],
                           summary_text: str,
                           model_used: str,
                           start_time: float) -> Dict[str, Any]:
        latency = time.time() - start_time
        
        return {
            "cluster_id": cluster.get("id"),
            "topic": cluster.get("topic"),
            "summary": summary_text,
            "model_used": model_used,
            "article_count": len(cluster.get("articles", [])),
            "keywords": cluster.get("keywords", []),
            "sources": cluster.get("sources", []),
            "latency": latency,
            "created_at": datetime.utcnow(),
            "word_count": len(summary_text.split()),
            "quality_score": self._estimate_quality_score(summary_text, cluster)
        }
        
    def _create_empty_summary(self, cluster: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "cluster_id": cluster.get("id"),
            "topic": cluster.get("topic", "unknown"),
            "summary": "No summary available",
            "model_used": "none",
            "article_count": len(cluster.get("articles", [])),
            "keywords": cluster.get("keywords", []),
            "sources": cluster.get("sources", []),
            "latency": 0.0,
            "created_at": datetime.utcnow(),
            "word_count": 0,
            "quality_score": 0.0,
            "error": True
        }
        
    def _estimate_quality_score(self, summary: str, cluster: Dict[str, Any]) -> float:
        try:
            # Simple quality metrics
            score = 0.0
            
            # Length appropriateness (0.3 weight)
            word_count = len(summary.split())
            if self.config.MIN_SUMMARY_LENGTH <= word_count <= self.config.MAX_SUMMARY_LENGTH:
                score += 0.3
            elif word_count > 0:
                score += 0.1
                
            # Keyword coverage (0.4 weight)
            keywords = cluster.get("keywords", [])
            if keywords:
                summary_lower = summary.lower()
                covered_keywords = sum(1 for kw in keywords if kw.lower() in summary_lower)
                score += (covered_keywords / len(keywords)) * 0.4
                
            # Basic coherence check (0.3 weight)
            sentences = self._split_sentences(summary)
            if len(sentences) >= 2:
                score += 0.3
            elif len(sentences) == 1 and len(summary.split()) > 10:
                score += 0.2
                
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error estimating quality score: {e}")
            return 0.5
            
    def _update_model_performance(self, model_name: str, start_time: float, success: bool):
        if model_name not in self.model_performance:
            return
            
        perf = self.model_performance[model_name]
        
        # Update request counts
        perf["total_requests"] += 1
        if not success:
            perf["failed_requests"] += 1
            
        # Update success rate
        perf["success_rate"] = 1.0 - (perf["failed_requests"] / perf["total_requests"])
        
        # Update average latency (only for successful requests)
        if success:
            latency = time.time() - start_time
            current_avg = perf["avg_latency"]
            successful_requests = perf["total_requests"] - perf["failed_requests"]
            
            if successful_requests == 1:
                perf["avg_latency"] = latency
            else:
                # Exponential moving average
                alpha = 0.1
                perf["avg_latency"] = alpha * latency + (1 - alpha) * current_avg
                
    async def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models"""
        status = {
            "available_models": list(self.models.keys()),
            "device": str(self.device),
            "performance": self.model_performance
        }
        
        return status
        
    async def cleanup(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
            
        # Clear model cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Summarization models cleaned up")
