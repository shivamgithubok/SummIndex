import re
import logging
import hashlib
from typing import List, Dict, Any, Set, Optional
from datetime import datetime
import unicodedata
import langdetect
from collections import defaultdict

logger = logging.getLogger(__name__)

class TextProcessor:    
    def __init__(self, config: Any):
        self.config = config
        self.seen_hashes: Set[str] = set()
        self.url_patterns = self._compile_url_patterns()
        self.cleaning_patterns = self._compile_cleaning_patterns()
        
    def _compile_url_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for URL detection"""
        patterns = [
            re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            re.compile(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
        ]
        return patterns
        
    def _compile_cleaning_patterns(self) -> List[Dict[str, Any]]:
        """Compile regex patterns for text cleaning"""
        patterns = [
            {"pattern": re.compile(r'\s+'), "replacement": " "},  # Multiple whitespace
            {"pattern": re.compile(r'\n+'), "replacement": " "},  # Multiple newlines
            {"pattern": re.compile(r'\t+'), "replacement": " "},  # Tabs
            {"pattern": re.compile(r'[^\w\s\.\,\!\?\:\;\-\(\)]'), "replacement": ""},  # Special chars
            {"pattern": re.compile(r'\.{2,}'), "replacement": "."},  # Multiple periods
            {"pattern": re.compile(r'\s*\.\s*'), "replacement": ". "},  # Period spacing
            {"pattern": re.compile(r'\s*,\s*'), "replacement": ", "},  # Comma spacing
        ]
        return patterns
        
    async def process_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        processed_articles = []
        
        for article in articles:
            try:
                processed_article = await self.process_single_article(article)
                if processed_article:
                    processed_articles.append(processed_article)
                    
            except Exception as e:
                logger.error(f"Error processing article {article.get('id', 'unknown')}: {e}")
                continue
                
        logger.info(f"Processed {len(processed_articles)} out of {len(articles)} articles")
        return processed_articles
        
    async def process_single_article(self, article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            # Extract and clean text fields
            title = self.clean_text(article.get("title", ""))
            description = self.clean_text(article.get("description", ""))
            content = self.clean_text(article.get("content", ""))
            
            # Validate minimum requirements
            if not title or len(title.split()) < 3:
                logger.debug("Skipping article with insufficient title")
                return None
                
            # Combine text for full processing
            full_text = f"{title}. {description}"
            if content and content != description:
                full_text += f" {content}"
                
            full_text = self.clean_text(full_text)
            
            # Check for duplicates
            content_hash = self.generate_content_hash(full_text)
            if content_hash in self.seen_hashes:
                logger.debug(f"Duplicate content detected: {content_hash}")
                return None
                
            self.seen_hashes.add(content_hash)
            
            # Detect language
            language = self.detect_language(full_text)
            if language not in self.config.NEWS_LANGUAGES:
                logger.debug(f"Skipping article in language: {language}")
                return None
                
            # Extract metadata
            word_count = len(full_text.split())
            sentence_count = len(self.split_sentences(full_text))
            
            processed_article = {
                **article,  # Keep original fields
                "title_clean": title,
                "description_clean": description,
                "content_clean": content,
                "full_text_clean": full_text,
                "content_hash": content_hash,
                "language_detected": language,
                "word_count": word_count,
                "sentence_count": sentence_count,
                "processed_at": datetime.utcnow(),
                "is_valid": True
            }
            
            return processed_article
            
        except Exception as e:
            logger.error(f"Error processing single article: {e}")
            return None
            
    def clean_text(self, text: str) -> str:
        if not text:
            return ""
            
        # Remove URLs
        for pattern in self.url_patterns:
            text = pattern.sub("", text)
            
        # Normalize unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Apply cleaning patterns
        for pattern_info in self.cleaning_patterns:
            text = pattern_info["pattern"].sub(pattern_info["replacement"], text)
            
        # Remove extra whitespace and strip
        text = text.strip()
        
        return text
        
    def generate_content_hash(self, text: str) -> str:
        # Normalize text for hashing (remove case, extra spaces)
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()
        
    def detect_language(self, text: str) -> str:
        try:
            if len(text) < 10:
                return "unknown"
                
            return langdetect.detect(text)
            
        except Exception as e:
            logger.debug(f"Language detection failed: {e}")
            return "unknown"
            
    def split_sentences(self, text: str) -> List[str]:
        # Simple sentence splitting - can be enhanced with NLTK/spaCy
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
        
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        # Simple keyword extraction based on word frequency
        # In production, use TF-IDF or more advanced methods
        
        # Common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter stop words and count frequency
        word_freq = defaultdict(int)
        for word in words:
            if word not in stop_words:
                word_freq[word] += 1
                
        # Sort by frequency and return top keywords
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in keywords[:max_keywords]]
        
    def validate_article_quality(self, article: Dict[str, Any]) -> bool:
        try:
            full_text = article.get("full_text_clean", "")
            word_count = article.get("word_count", 0)
            
            # Minimum word count
            if word_count < 20:
                return False
                
            # Check for reasonable title length
            title = article.get("title_clean", "")
            if len(title.split()) < 3 or len(title.split()) > 50:
                return False
                
            # Check for spam indicators
            if self._contains_spam_indicators(full_text):
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating article quality: {e}")
            return False
            
    def _contains_spam_indicators(self, text: str) -> bool:
        """Check for spam indicators in text"""
        spam_indicators = [
            r'click here',
            r'free money',
            r'make.*money.*fast',
            r'weight.*loss',
            r'viagra',
            r'lottery.*winner'
        ]
        
        text_lower = text.lower()
        for pattern in spam_indicators:
            if re.search(pattern, text_lower):
                return True
                
        return False
        
    def cleanup_memory(self, max_hashes: int = 100000):
        if len(self.seen_hashes) > max_hashes:
            # Keep only recent hashes (simple FIFO)
            self.seen_hashes = set(list(self.seen_hashes)[-max_hashes//2:])
            logger.info(f"Cleaned up hash memory, keeping {len(self.seen_hashes)} hashes")
