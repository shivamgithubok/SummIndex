import asyncio
import aiohttp
import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from config import Config

logger = logging.getLogger(__name__)

class RealGNewsClient:
    """Real GNews API client for fetching authentic news data"""
    
    def __init__(self, api_key: str, config: Config):
        self.api_key = api_key
        self.config = config
        self.base_url = "https://gnews.io/api/v4"
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_fetch_time = 0
        self.rate_limit_delay = 60  # 1 minute between requests for free tier
        
    async def initialize(self):
        """Initialize the HTTP session"""
        self.session = aiohttp.ClientSession()
        logger.info("Real GNews client initialized")
        
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            
    async def fetch_top_headlines(self, 
                                 category: str = "general",
                                 country: str = "us",
                                 max_articles: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch top headlines from GNews API
        
        Args:
            category: News category (general, world, nation, business, technology, entertainment, sports, science, health)
            country: Country code (us, gb, ca, au, etc.)
            max_articles: Maximum number of articles to fetch
            
        Returns:
            List of article dictionaries
        """
        if not self.session:
            await self.initialize()
            
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_fetch_time < self.rate_limit_delay:
            wait_time = self.rate_limit_delay - (current_time - self.last_fetch_time)
            logger.info(f"Rate limiting: waiting {wait_time:.1f} seconds")
            await asyncio.sleep(wait_time)
            
        try:
            url = f"{self.base_url}/top-headlines"
            params = {
                "token": self.api_key,
                "category": category,
                "country": country,
                "max": min(max_articles, 10),  # API limit
                "lang": "en"
            }
            
            logger.info(f"Fetching headlines: category={category}, country={country}")
            
            async with self.session.get(url, params=params) as response:
                self.last_fetch_time = time.time()
                
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"GNews API error {response.status}: {error_text}")
                    return []
                    
                data = await response.json()
                
                if data.get("totalArticles", 0) == 0:
                    logger.warning("No articles returned from GNews API")
                    return []
                    
                articles = data.get("articles", [])
                processed_articles = []
                
                for article in articles:
                    processed_article = {
                        "id": f"gnews_{hash(article.get('url', ''))%100000}",
                        "title": article.get("title", ""),
                        "content": article.get("content", article.get("description", "")),
                        "url": article.get("url", ""),
                        "published_at": article.get("publishedAt", datetime.now().isoformat()),
                        "source": article.get("source", {}).get("name", "Unknown"),
                        "image_url": article.get("image", ""),
                        "category": category,
                        "country": country
                    }
                    processed_articles.append(processed_article)
                    
                logger.info(f"Successfully fetched {len(processed_articles)} real articles")
                return processed_articles
                
        except Exception as e:
            logger.error(f"Error fetching from GNews API: {e}")
            return []
            
    async def search_news(self, 
                         query: str,
                         sortby: str = "relevance",
                         max_articles: int = 10) -> List[Dict[str, Any]]:
        """
        Search for news articles by query
        
        Args:
            query: Search query
            sortby: Sort order (relevance, publishedAt)
            max_articles: Maximum number of articles
            
        Returns:
            List of article dictionaries
        """
        if not self.session:
            await self.initialize()
            
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_fetch_time < self.rate_limit_delay:
            wait_time = self.rate_limit_delay - (current_time - self.last_fetch_time)
            await asyncio.sleep(wait_time)
            
        try:
            url = f"{self.base_url}/search"
            params = {
                "token": self.api_key,
                "q": query,
                "sortby": sortby,
                "max": min(max_articles, 10),
                "lang": "en"
            }
            
            logger.info(f"Searching news: query='{query}', sortby={sortby}")
            
            async with self.session.get(url, params=params) as response:
                self.last_fetch_time = time.time()
                
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"GNews search API error {response.status}: {error_text}")
                    return []
                    
                data = await response.json()
                articles = data.get("articles", [])
                
                processed_articles = []
                for article in articles:
                    processed_article = {
                        "id": f"gnews_search_{hash(article.get('url', ''))%100000}",
                        "title": article.get("title", ""),
                        "content": article.get("content", article.get("description", "")),
                        "url": article.get("url", ""),
                        "published_at": article.get("publishedAt", datetime.now().isoformat()),
                        "source": article.get("source", {}).get("name", "Unknown"),
                        "image_url": article.get("image", ""),
                        "query": query
                    }
                    processed_articles.append(processed_article)
                    
                logger.info(f"Search returned {len(processed_articles)} articles")
                return processed_articles
                
        except Exception as e:
            logger.error(f"Error searching GNews API: {e}")
            return []

class EnhancedNewsProcessor:
    """Enhanced news processing with real data integration"""
    
    def __init__(self, config: Config):
        self.config = config
        self.gnews_client: Optional[RealGNewsClient] = None
        self.use_real_data = False
        
    async def initialize(self):
        """Initialize with real GNews client if API key is available"""
        api_key = self.config.GNEWS_API_KEY
        
        # Check if we have a real API key (not the default placeholder)
        if api_key and api_key != "default_gnews_key" and len(api_key) > 10:
            try:
                self.gnews_client = RealGNewsClient(api_key, self.config)
                await self.gnews_client.initialize()
                
                # Test the API with a simple request
                test_articles = await self.gnews_client.fetch_top_headlines(max_articles=1)
                if test_articles:
                    self.use_real_data = True
                    logger.info("✅ Successfully connected to GNews API - using REAL news data!")
                else:
                    logger.warning("⚠️ GNews API key present but no data returned")
                    
            except Exception as e:
                logger.error(f"❌ Failed to initialize GNews API: {e}")
                logger.info("💡 Falling back to sample data - provide a valid GNews API key for real news")
        else:
            logger.info("🔑 No GNews API key provided - using sample data")
            logger.info("💡 Set GNEWS_API_KEY environment variable to use real news data")
            
    async def fetch_diverse_news(self, max_articles: int = 20) -> List[Dict[str, Any]]:
        """Fetch diverse news from multiple categories"""
        if not self.use_real_data or not self.gnews_client:
            return self._generate_sample_articles()
            
        try:
            all_articles = []
            categories = ["general", "technology", "business", "health", "science"]
            articles_per_category = max(1, max_articles // len(categories))
            
            for category in categories:
                try:
                    articles = await self.gnews_client.fetch_top_headlines(
                        category=category,
                        max_articles=articles_per_category
                    )
                    all_articles.extend(articles)
                    
                    # Add small delay between category requests
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error fetching {category} news: {e}")
                    continue
                    
            if not all_articles:
                logger.warning("No real articles fetched, falling back to samples")
                return self._generate_sample_articles()
                
            logger.info(f"📰 Fetched {len(all_articles)} REAL news articles from GNews API")
            return all_articles[:max_articles]
            
        except Exception as e:
            logger.error(f"Error in diverse news fetch: {e}")
            return self._generate_sample_articles()
            
    def _generate_sample_articles(self) -> List[Dict[str, Any]]:
        """Generate sample articles for demonstration"""
        current_time = datetime.now().isoformat()
        
        sample_articles = [
            {
                "id": f"sample_{int(time.time())}",
                "title": "AI Breakthrough: New Language Model Achieves Human-Level Performance",
                "content": "Researchers have developed a revolutionary AI system that demonstrates human-level understanding across multiple domains. The breakthrough could transform industries from healthcare to education, offering unprecedented capabilities in natural language processing and reasoning.",
                "url": "https://example.com/ai-breakthrough",
                "published_at": current_time,
                "source": "Tech Innovation Daily",
                "category": "technology",
                "image_url": "",
                "is_sample": True
            },
            {
                "id": f"sample_{int(time.time()) + 1}",
                "title": "Global Climate Summit: World Leaders Commit to Ambitious Carbon Reduction Targets",
                "content": "In a historic agreement at the Global Climate Summit, world leaders have committed to reducing carbon emissions by 50% within the next decade. The comprehensive plan includes investments in renewable energy, sustainable transportation, and green technology innovation.",
                "url": "https://example.com/climate-summit",
                "published_at": current_time,
                "source": "International News Network",
                "category": "general",
                "image_url": "",
                "is_sample": True
            },
            {
                "id": f"sample_{int(time.time()) + 2}",
                "title": "Medical Breakthrough: Gene Therapy Shows Promise for Rare Diseases",
                "content": "Scientists have achieved remarkable success in treating rare genetic disorders using advanced gene therapy techniques. Clinical trials show significant improvement in patient outcomes, offering hope for millions affected by previously incurable conditions.",
                "url": "https://example.com/gene-therapy",
                "published_at": current_time,
                "source": "Medical Research Today",
                "category": "health",
                "image_url": "",
                "is_sample": True
            }
        ]
        
        logger.info(f"📝 Generated {len(sample_articles)} sample articles for demonstration")
        return sample_articles
        
    async def cleanup(self):
        """Cleanup resources"""
        if self.gnews_client:
            await self.gnews_client.cleanup()