import asyncio
import aiohttp
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import json

logger = logging.getLogger(__name__)

class GNewsClient:    
    def __init__(self, api_key: str, config: Any):
        self.api_key = api_key
        self.config = config
        self.base_url = "https://gnews.io/api/v4"
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_fetch_time: Optional[datetime] = None
        
    async def initialize(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                "User-Agent": "SummIndex/1.0.0"
            }
        )
        logger.info("GNews client initialized")
        
    async def cleanup(self):
        if self.session:
            await self.session.close()
            
    async def fetch_news(self, 
                        query: Optional[str] = None,
                        sources: Optional[List[str]] = None,
                        lang: str = "en",
                        country: str = "us",
                        max_articles: int = 50) -> List[Dict[str, Any]]:
        if not self.session:
            await self.initialize()
            
        try:
            # Build API parameters
            params = {
                "apikey": self.api_key,
                "lang": lang,
                "country": country,
                "max": min(max_articles, 100)  # API limit
            }
            
            if query:
                params["q"] = query
                endpoint = f"{self.base_url}/search"
            else:
                endpoint = f"{self.base_url}/top-headlines"
                
            if sources:
                params["in"] = ",".join(sources)
                
            # Add time filter for incremental updates
            if self.last_fetch_time:
                params["from"] = self.last_fetch_time.isoformat()
                
            logger.debug(f"Fetching news with params: {params}")
            
            async with self.session.get(endpoint, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = data.get("articles", [])
                    
                    # Process and normalize articles
                    processed_articles = []
                    for article in articles:
                        processed_article = self._process_article(article)
                        if processed_article:
                            processed_articles.append(processed_article)
                            
                    self.last_fetch_time = datetime.now(timezone.utc)
                    logger.info(f"Fetched {len(processed_articles)} articles")
                    
                    return processed_articles
                    
                elif response.status == 429:
                    logger.warning("Rate limit exceeded, waiting...")
                    await asyncio.sleep(60)
                    return []
                    
                else:
                    logger.error(f"API error: {response.status} - {await response.text()}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []
            
    def _process_article(self, article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            # Extract required fields
            title = article.get("title", "").strip()
            description = article.get("description", "").strip()
            content = article.get("content", "").strip()
            
            if not title or not description:
                logger.debug("Skipping article with missing title or description")
                return None
                
            # Combine title, description, and content
            full_text = f"{title}. {description}"
            if content and content != description:
                full_text += f" {content}"
                
            processed = {
                "id": self._generate_article_id(article),
                "title": title,
                "description": description,
                "content": content,
                "full_text": full_text,
                "url": article.get("url", ""),
                "source": article.get("source", {}).get("name", "Unknown"),
                "published_at": self._parse_datetime(article.get("publishedAt")),
                "fetched_at": datetime.now(timezone.utc),
                "image": article.get("image"),
                "language": "en"  # Assume English for now
            }
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing article: {e}")
            return None
            
    def _generate_article_id(self, article: Dict[str, Any]) -> str:
        import hashlib
        
        # Use URL if available, otherwise use title + source
        identifier = article.get("url")
        if not identifier:
            title = article.get("title", "")
            source = article.get("source", {}).get("name", "")
            identifier = f"{title}_{source}"
            
        return hashlib.md5(identifier.encode()).hexdigest()
        
    def _parse_datetime(self, dt_string: Optional[str]) -> Optional[datetime]:
        if not dt_string:
            return None
            
        try:
            # Handle ISO format with timezone
            if dt_string.endswith('Z'):
                dt_string = dt_string[:-1] + '+00:00'
                
            return datetime.fromisoformat(dt_string)
            
        except Exception as e:
            logger.error(f"Error parsing datetime '{dt_string}': {e}")
            return None
            
    async def fetch_trending_topics(self, 
                                  lang: str = "en",
                                  country: str = "us") -> List[str]:
        try:
            # For now, return a static list of common news topics
            # In a production system, this could be derived from current headlines
            trending_topics = [
                "politics", "economy", "technology", "health", "sports", 
                "entertainment", "science", "climate", "business", "education"
            ]
            
            return trending_topics
            
        except Exception as e:
            logger.error(f"Error fetching trending topics: {e}")
            return []
            
    async def stream_news(self, 
                         interval: int = 30,
                         **kwargs) -> List[Dict[str, Any]]:
        try:
            articles = await self.fetch_news(**kwargs)
            
            # Schedule next fetch
            await asyncio.sleep(interval)
            
            return articles
            
        except Exception as e:
            logger.error(f"Error in news stream: {e}")
            return []
