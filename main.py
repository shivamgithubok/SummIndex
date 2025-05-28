"""
SummIndex - Unified Real-Time News Summarization System
Streamlined with LangChain, Real APIs, and Comprehensive Evaluation
"""
import asyncio
from typing import Dict, Any
from datetime import datetime
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn

from core import SummIndexCore
from logger import logger

load_dotenv()

class SummIndexApp:
    """Unified SummIndex Application"""
    
    def __init__(self):
        self.core = SummIndexCore()
        self.app = FastAPI(
            title="SummIndex - Real-Time News Summarization",
            description="Advanced news summarization with LangChain and real APIs",
            version="2.0.0"
        )
        self.templates = Jinja2Templates(directory="templates")
        self._setup_middleware()
        self._setup_routes()
        
    def _setup_middleware(self):
        """Setup CORS middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
    def _setup_routes(self):
        """Setup all routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard(request: Request):
            """Main dashboard"""
            return self.templates.TemplateResponse("dashboard.html", {"request": request})
            
        @self.app.get("/evaluation", response_class=HTMLResponse)
        async def evaluation_page(request: Request):
            """Evaluation dashboard"""
            return self.templates.TemplateResponse("evaluation.html", {"request": request})
            
        @self.app.get("/analysis", response_class=HTMLResponse)
        async def analysis_page(request: Request):
            """Performance analysis interface"""
            return self.templates.TemplateResponse("analysis.html", {"request": request})
            
        @self.app.get("/api/info")
        async def api_info():
            """API information"""
            return {
                "service": "SummIndex",
                "version": "2.0.0",
                "status": "running",
                "features": ["Real APIs", "LangChain", "Performance Analysis"],
                "timestamp": datetime.now().isoformat()
            }
            
        @self.app.get("/api/health")
        async def health_check():
            """Health check"""
            return {
                "status": "healthy",
                "pipeline_running": self.core.running,
                "total_summaries": len(self.core.summaries),
                "cycles_completed": self.core.cycle_count,
                "timestamp": datetime.now().isoformat()
            }
            
        @self.app.get("/api/summaries")
        async def get_summaries():
            """Get recent summaries"""
            summaries = list(self.core.summaries.values())
            summaries.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            
            return {
                "summaries": summaries[:20],  # Latest 20
                "total_count": len(summaries),
                "timestamp": datetime.now().isoformat()
            }
            
        @self.app.get("/api/evaluation")
        async def get_evaluation():
            """Get evaluation data for analysis interface"""
            return self.core.get_evaluation_data()
            
        @self.app.post("/api/search")
        async def search_summaries(request: dict):
            """Search summaries"""
            query = request.get("query", "")
            if not query:
                raise HTTPException(status_code=400, detail="Query required")
                
            results = await self.core.search(query)
            return {
                "results": results[:10],  # Top 10 results
                "total_found": len(results),
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
            
        @self.app.get("/api/stats")
        async def get_stats():
            """Get system statistics"""
            stats = self.core.tracker.get_stats()
            return {
                "performance": stats,
                "system": {
                    "total_summaries": len(self.core.summaries),
                    "pipeline_running": self.core.running,
                    "cycles_completed": self.core.cycle_count
                },
                "timestamp": datetime.now().isoformat()
            }
            
    async def start(self):
        """Start the application"""
        try:
            logger.info("🚀 Starting SummIndex System...")
            
            # Start core processing in background
            processing_task = asyncio.create_task(self.core.start_processing())
            
            # Start web server
            config = uvicorn.Config(
                self.app,
                host="0.0.0.0",
                port=5000,
                log_level="info"
            )
            server = uvicorn.Server(config)
            
            logger.info("✅ SummIndex running on http://0.0.0.0:5000")
            logger.info("📊 Dashboard: http://0.0.0.0:5000/")
            logger.info("📈 Analysis: http://0.0.0.0:5000/analysis")
            
            await server.serve()
            
        except Exception as e:
            logger.error(f"Failed to start: {e}")
            raise
            
    def stop(self):
        """Stop the application"""
        self.core.stop_processing()
        logger.info("👋 SummIndex stopped")

async def main():
    """Main entry point"""
    app = SummIndexApp()
    
    try:
        await app.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        app.stop()

if __name__ == "__main__":
    asyncio.run(main())