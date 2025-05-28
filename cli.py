#!/usr/bin/env python3
"""
Command Line Interface for SummIndex
"""

import asyncio
import argparse
import logging
import sys
import json
from datetime import datetime
from typing import Optional

from summindex.core.pipeline import SummIndexPipeline
from config import Config

def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

async def run_pipeline(config: Config, duration: Optional[int] = None):
    """Run the main SummIndex pipeline"""
    pipeline = SummIndexPipeline(config)
    
    try:
        await pipeline.initialize()
        
        if duration:
            print(f"Running pipeline for {duration} seconds...")
            # Run for specified duration
            pipeline_task = asyncio.create_task(pipeline.run())
            await asyncio.sleep(duration)
            await pipeline.stop()
            await pipeline_task
        else:
            print("Starting pipeline (Ctrl+C to stop)...")
            await pipeline.run()
            
    except KeyboardInterrupt:
        print("\nStopping pipeline...")
    finally:
        await pipeline.cleanup()

async def search_content(config: Config, query: str, search_type: str = "hybrid"):
    """Search for content using the pipeline"""
    pipeline = SummIndexPipeline(config)
    
    try:
        await pipeline.initialize()
        
        print(f"Searching for: '{query}' (type: {search_type})")
        results = await pipeline.search(query, search_type)
        
        if results:
            print(f"\nFound {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result.get('title', 'No title')}")
                print(f"   Type: {result.get('type', 'unknown')}")
                print(f"   Score: {result.get('_score', result.get('combined_score', 'N/A'))}")
                if result.get('summary'):
                    print(f"   Summary: {result['summary'][:100]}...")
                    
        else:
            print("No results found.")
            
    except Exception as e:
        print(f"Search failed: {e}")
    finally:
        await pipeline.cleanup()

async def show_status(config: Config):
    """Show pipeline status"""
    pipeline = SummIndexPipeline(config)
    
    try:
        await pipeline.initialize()
        status = await pipeline.get_pipeline_status()
        
        print("SummIndex Pipeline Status")
        print("=" * 40)
        
        # Component status
        print("\nComponent Status:")
        for component, is_healthy in status.get("component_status", {}).items():
            status_str = "✓ Online" if is_healthy else "✗ Offline"
            print(f"  {component}: {status_str}")
            
        # Performance metrics
        perf_metrics = status.get("performance_metrics", {})
        print(f"\nPerformance Metrics:")
        print(f"  Average Latency: {perf_metrics.get('avg_latency', 0):.2f}s")
        print(f"  Average Quality: {perf_metrics.get('avg_quality', 0):.2f}")
        print(f"  Average Accuracy: {perf_metrics.get('avg_accuracy', 0):.3f}")
        print(f"  Latency Target Met: {'Yes' if perf_metrics.get('latency_target_met', 0) else 'No'}")
        
        # Memory statistics
        memory_stats = status.get("memory_statistics", {})
        print(f"\nMemory Statistics:")
        print(f"  Total Memories: {memory_stats.get('total_memories', 0)}")
        print(f"  Unique Topics: {memory_stats.get('unique_topics', 0)}")
        print(f"  Memory Utilization: {memory_stats.get('memory_utilization', 0):.1%}")
        
        # Pipeline statistics
        pipeline_stats = status.get("pipeline_statistics", {})
        print(f"\nPipeline Statistics:")
        print(f"  Running: {'Yes' if pipeline_stats.get('running', False) else 'No'}")
        print(f"  Current Summaries: {pipeline_stats.get('current_summaries', 0)}")
        
    except Exception as e:
        print(f"Failed to get status: {e}")
    finally:
        await pipeline.cleanup()

async def train_rl_agent(config: Config, episodes: int = 1000):
    """Train the RL agent"""
    from summindex.rl_agent.summarization_agent import SummarizationAgent
    
    agent = SummarizationAgent(config)
    
    try:
        await agent.initialize()
        print(f"Training RL agent for {episodes} episodes...")
        await agent.train_agent(episodes)
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed: {e}")
    finally:
        await agent.cleanup()

async def evaluate_system(config: Config):
    """Generate evaluation report"""
    from summindex.evaluation.metrics import EvaluationMetrics
    
    metrics = EvaluationMetrics(config)
    
    try:
        await metrics.initialize()
        
        print("Generating evaluation report...")
        report = await metrics.generate_evaluation_report()
        
        print("\nEvaluation Report")
        print("=" * 40)
        
        # Performance metrics
        perf_metrics = report.get("performance_metrics", {})
        print(f"\nPerformance Metrics:")
        for metric, value in perf_metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: {value}")
                
        # Targets met
        targets = report.get("targets_met", {})
        print(f"\nTargets Met:")
        for target, met in targets.items():
            status = "✓ Yes" if met > 0.8 else "✗ No"
            print(f"  {target}: {status} ({met:.1%})")
            
        # Recommendations
        recommendations = report.get("recommendations", [])
        print(f"\nRecommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
            
    except Exception as e:
        print(f"Evaluation failed: {e}")
    finally:
        await metrics.cleanup()

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="SummIndex CLI")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run pipeline
    run_parser = subparsers.add_parser("run", help="Run the SummIndex pipeline")
    run_parser.add_argument("--duration", type=int, help="Run duration in seconds (default: unlimited)")
    
    # Search
    search_parser = subparsers.add_parser("search", help="Search for content")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--type", default="hybrid", choices=["text", "semantic", "hybrid"],
                              help="Search type")
    
    # Status
    subparsers.add_parser("status", help="Show pipeline status")
    
    # Train RL agent
    train_parser = subparsers.add_parser("train", help="Train the RL agent")
    train_parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    
    # Evaluate
    subparsers.add_parser("evaluate", help="Generate evaluation report")
    
    # Config test
    subparsers.add_parser("config", help="Test configuration")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load configuration
    config = Config()
    
    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please check your environment variables.")
        sys.exit(1)
    
    # Execute command
    if args.command == "run":
        asyncio.run(run_pipeline(config, args.duration))
    elif args.command == "search":
        asyncio.run(search_content(config, args.query, args.type))
    elif args.command == "status":
        asyncio.run(show_status(config))
    elif args.command == "train":
        asyncio.run(train_rl_agent(config, args.episodes))
    elif args.command == "evaluate":
        asyncio.run(evaluate_system(config))
    elif args.command == "config":
        print("Configuration Test")
        print("=" * 40)
        print(f"GNews API Key: {'Set' if config.GNEWS_API_KEY != 'default_gnews_key' else 'Not set'}")
        print(f"Elasticsearch Host: {config.ELASTICSEARCH_HOST}:{config.ELASTICSEARCH_PORT}")
        print(f"Target Latency: {config.TARGET_LATENCY}s")
        print(f"Target Accuracy: {config.TARGET_ACCURACY}")
        print(f"Models: {list(config.SUMMARIZATION_MODELS.keys())}")
        print("Configuration is valid!")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
