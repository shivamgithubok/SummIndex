import logging
import numpy as np
import asyncio
import pickle
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
from gym import spaces

logger = logging.getLogger(__name__)

class SummarizationEnvironment(gym.Env):    
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        
        # Action space: [summarize_now, wait, prioritize_topic]
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [cluster_size, time_since_last_update, topic_importance, 
        #                    user_interest, content_novelty, system_load]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )
        
        # Environment state
        self.current_clusters: Dict[str, Dict[str, Any]] = {}
        self.user_feedback: Dict[str, float] = {}
        self.system_metrics: Dict[str, float] = {}
        self.episode_step = 0
        self.max_episode_steps = 100
        
    def reset(self) -> np.ndarray:
        self.current_clusters.clear()
        self.user_feedback.clear()
        self.episode_step = 0
        
        # Initialize with dummy observation
        return np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.episode_step += 1
        
        # Calculate reward based on action and current state
        reward = self._calculate_reward(action)
        
        # Update state based on action
        next_state = self._get_next_state(action)
        
        # Check if episode is done
        done = self.episode_step >= self.max_episode_steps
        
        info = {
            "action": action,
            "step": self.episode_step,
            "reward": reward
        }
        
        return next_state, reward, done, info
        
    def _calculate_reward(self, action: int) -> float:
        reward = 0.0
        
        # Reward for timely summarization
        if action == 0:  # summarize_now
            # Positive reward for summarizing when cluster is ready
            cluster_readiness = self._get_cluster_readiness()
            reward += cluster_readiness * 10.0
            
            # Penalty for too frequent summarization
            if self._is_too_frequent():
                reward -= 5.0
                
        elif action == 1:  # wait
            # Small positive reward for waiting when cluster is not ready
            cluster_readiness = self._get_cluster_readiness()
            reward += (1.0 - cluster_readiness) * 2.0
            
            # Penalty for waiting too long
            if self._waited_too_long():
                reward -= 3.0
                
        elif action == 2:  # prioritize_topic
            # Reward for prioritizing important topics
            topic_importance = self._get_topic_importance()
            reward += topic_importance * 5.0
            
        # Additional rewards based on system metrics
        latency_penalty = -max(0, (self.system_metrics.get("latency", 0) - 2.0) * 5.0)
        quality_bonus = self.system_metrics.get("quality_score", 0.5) * 3.0
        
        reward += latency_penalty + quality_bonus
        
        return reward
        
    def _get_cluster_readiness(self) -> float:
        if not self.current_clusters:
            return 0.0
            
        readiness_scores = []
        for cluster in self.current_clusters.values():
            size = cluster.get("size", 0)
            age = cluster.get("age_minutes", 0)
            
            # Higher readiness for larger, older clusters
            size_score = min(1.0, size / 10.0)
            age_score = min(1.0, age / 30.0)  # 30 minutes max
            
            readiness_scores.append((size_score + age_score) / 2.0)
            
        return np.mean(readiness_scores)
        
    def _get_topic_importance(self) -> float:
        if not self.current_clusters:
            return 0.0
            
        importance_scores = []
        for cluster in self.current_clusters.values():
            # Factor in user interest, news impact, etc.
            user_interest = self.user_feedback.get(cluster.get("topic", ""), 0.5)
            source_diversity = len(cluster.get("sources", [])) / 5.0  # Normalize by 5 sources
            
            importance = (user_interest + source_diversity) / 2.0
            importance_scores.append(importance)
            
        return np.mean(importance_scores)
        
    def _is_too_frequent(self) -> bool:
        return self.system_metrics.get("summarization_frequency", 0) > 0.5
        
    def _waited_too_long(self) -> bool:
        return self.system_metrics.get("time_since_last_summary", 0) > 600  # 10 minutes
        
    def _get_next_state(self, action: int) -> np.ndarray:
        """Get next state observation"""
        # This would be updated with real system state
        # For now, return a simulated state
        cluster_size = len(self.current_clusters) / 50.0  # Normalize by max clusters
        time_factor = min(1.0, self.episode_step / self.max_episode_steps)
        topic_importance = self._get_topic_importance()
        user_interest = np.mean(list(self.user_feedback.values())) if self.user_feedback else 0.5
        content_novelty = self.system_metrics.get("novelty_score", 0.5)
        system_load = self.system_metrics.get("cpu_usage", 0.3)
        
        return np.array([
            cluster_size, time_factor, topic_importance,
            user_interest, content_novelty, system_load
        ], dtype=np.float32)
        
    def update_clusters(self, clusters: Dict[str, Dict[str, Any]]):
        """Update current clusters"""
        self.current_clusters = clusters
        
    def update_metrics(self, metrics: Dict[str, float]):
        """Update system metrics"""
        self.system_metrics.update(metrics)
        
    def add_user_feedback(self, topic: str, rating: float):
        """Add user feedback for topics"""
        self.user_feedback[topic] = rating

class SummarizationAgent:
    """RL agent for summarization decision making"""
    
    def __init__(self, config: Any):
        self.config = config
        self.env: Optional[SummarizationEnvironment] = None
        self.model: Optional[PPO] = None
        self.training_data: deque = deque(maxlen=10000)
        self.episode_rewards: List[float] = []
        self.decision_history: List[Dict[str, Any]] = []
        self.user_preferences: Dict[str, float] = defaultdict(lambda: 0.5)
        
    async def initialize(self):
        """Initialize the RL agent"""
        try:
            logger.info("Initializing RL agent...")
            
            # Create environment
            self.env = SummarizationEnvironment(self.config)
            
            # Load or create model
            model_path = self.config.RL_MODEL_PATH
            if os.path.exists(f"{model_path}.zip"):
                logger.info(f"Loading existing RL model from {model_path}")
                self.model = PPO.load(f"{model_path}.zip", env=self.env)
            else:
                logger.info("Creating new RL model")
                self.model = PPO(
                    "MlpPolicy",
                    self.env,
                    verbose=1,
                    learning_rate=0.0003,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2
                )
                
            logger.info("RL agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RL agent: {e}")
            raise
            
    async def decide_summarization_action(self, 
                                        clusters: Dict[str, Dict[str, Any]],
                                        system_metrics: Dict[str, float]) -> Dict[str, Any]:
        try:
            if not self.model or not self.env:
                # Fallback to rule-based decisions
                return await self._fallback_decision(clusters, system_metrics)
                
            # Update environment with current state
            self.env.update_clusters(clusters)
            self.env.update_metrics(system_metrics)
            
            # Get current observation
            observation = self._create_observation(clusters, system_metrics)
            
            # Get action from model
            action, _states = self.model.predict(observation, deterministic=False)
            
            # Convert action to decision
            decision = self._action_to_decision(action, clusters, system_metrics)
            
            # Store decision for learning
            self.decision_history.append({
                "timestamp": datetime.utcnow(),
                "observation": observation,
                "action": action,
                "clusters": len(clusters),
                "decision": decision
            })
            
            return decision
            
        except Exception as e:
            logger.error(f"Error in RL decision making: {e}")
            return await self._fallback_decision(clusters, system_metrics)
            
    def _create_observation(self, 
                          clusters: Dict[str, Dict[str, Any]],
                          system_metrics: Dict[str, float]) -> np.ndarray:
        
        # Calculate cluster readiness
        cluster_readiness = 0.0
        if clusters:
            readiness_scores = []
            for cluster in clusters.values():
                size = cluster.get("size", 0)
                age_minutes = self._calculate_cluster_age(cluster)
                
                size_score = min(1.0, size / 10.0)
                age_score = min(1.0, age_minutes / 30.0)
                readiness_scores.append((size_score + age_score) / 2.0)
                
            cluster_readiness = np.mean(readiness_scores)
            
        # Time since last update
        time_factor = system_metrics.get("time_since_last_summary", 0) / 600.0  # Normalize by 10 minutes
        time_factor = min(1.0, time_factor)
        
        # Topic importance
        topic_importance = self._calculate_topic_importance(clusters)
        
        # User interest (based on stored preferences)
        user_interest = self._calculate_user_interest(clusters)
        
        # Content novelty
        content_novelty = system_metrics.get("novelty_score", 0.5)
        
        # System load
        system_load = system_metrics.get("cpu_usage", 0.3)
        
        observation = np.array([
            cluster_readiness,
            time_factor,
            topic_importance,
            user_interest,
            content_novelty,
            system_load
        ], dtype=np.float32)
        
        return observation
        
    def _calculate_cluster_age(self, cluster: Dict[str, Any]) -> float:
        created_at = cluster.get("created_at")
        if not created_at:
            return 0.0
            
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            except:
                return 0.0
        elif not isinstance(created_at, datetime):
            return 0.0
            
        age = datetime.utcnow() - created_at
        return age.total_seconds() / 60.0  # Convert to minutes
        
    def _calculate_topic_importance(self, clusters: Dict[str, Dict[str, Any]]) -> float:
        if not clusters:
            return 0.0
            
        importance_scores = []
        for cluster in clusters.values():
            topic = cluster.get("topic", "")
            sources_count = len(cluster.get("sources", []))
            article_count = cluster.get("size", 0)
            
            # Importance based on source diversity and article count
            source_factor = min(1.0, sources_count / 5.0)
            count_factor = min(1.0, article_count / 15.0)
            
            # Factor in user preferences
            user_pref = self.user_preferences.get(topic, 0.5)
            
            importance = (source_factor + count_factor + user_pref) / 3.0
            importance_scores.append(importance)
            
        return np.mean(importance_scores)
        
    def _calculate_user_interest(self, clusters: Dict[str, Dict[str, Any]]) -> float:
        if not clusters:
            return 0.5
            
        interest_scores = []
        for cluster in clusters.values():
            topic = cluster.get("topic", "")
            interest = self.user_preferences.get(topic, 0.5)
            interest_scores.append(interest)
            
        return np.mean(interest_scores)
        
    def _action_to_decision(self, 
                          action: int,
                          clusters: Dict[str, Dict[str, Any]],
                          system_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Convert RL action to summarization decision"""
        
        current_time = datetime.utcnow()
        
        if action == 0:  # Summarize now
            # Select clusters ready for summarization
            ready_clusters = self._select_ready_clusters(clusters)
            
            return {
                "action": "summarize",
                "clusters": ready_clusters,
                "priority": "immediate",
                "reason": "rl_agent_decision",
                "timestamp": current_time,
                "confidence": 0.8
            }
            
        elif action == 1:  # Wait
            return {
                "action": "wait",
                "wait_time": 60,  # Wait 1 minute
                "reason": "clusters_not_ready",
                "timestamp": current_time,
                "confidence": 0.7
            }
            
        elif action == 2:  # Prioritize topic
            # Find most important topic to prioritize
            priority_topic = self._find_priority_topic(clusters)
            
            return {
                "action": "prioritize",
                "topic": priority_topic,
                "boost_factor": 1.5,
                "reason": "high_importance_topic",
                "timestamp": current_time,
                "confidence": 0.6
            }
            
        else:
            # Default fallback
            return {
                "action": "wait",
                "wait_time": 30,
                "reason": "unknown_action",
                "timestamp": current_time,
                "confidence": 0.3
            }
            
    def _select_ready_clusters(self, clusters: Dict[str, Dict[str, Any]]) -> List[str]:
        """Select clusters ready for summarization"""
        ready_clusters = []
        
        for cluster_id, cluster in clusters.items():
            size = cluster.get("size", 0)
            age_minutes = self._calculate_cluster_age(cluster)
            
            # Consider ready if has enough articles and is old enough
            if size >= self.config.MIN_CLUSTER_SIZE and age_minutes >= 5:
                ready_clusters.append(cluster_id)
                
        return ready_clusters
        
    def _find_priority_topic(self, clusters: Dict[str, Dict[str, Any]]) -> str:
        """Find the most important topic to prioritize"""
        if not clusters:
            return "general"
            
        topic_scores = {}
        for cluster in clusters.values():
            topic = cluster.get("topic", "general")
            size = cluster.get("size", 0)
            sources = len(cluster.get("sources", []))
            user_pref = self.user_preferences.get(topic, 0.5)
            
            score = size * 0.4 + sources * 0.3 + user_pref * 0.3
            topic_scores[topic] = score
            
        return max(topic_scores.items(), key=lambda x: x[1])[0]
        
    async def _fallback_decision(self, 
                               clusters: Dict[str, Dict[str, Any]],
                               system_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Fallback rule-based decision making"""
        current_time = datetime.utcnow()
        
        # Simple rule-based logic
        ready_clusters = []
        for cluster_id, cluster in clusters.items():
            size = cluster.get("size", 0)
            age_minutes = self._calculate_cluster_age(cluster)
            
            if size >= self.config.MIN_CLUSTER_SIZE and age_minutes >= 10:
                ready_clusters.append(cluster_id)
                
        if ready_clusters:
            return {
                "action": "summarize",
                "clusters": ready_clusters,
                "priority": "normal",
                "reason": "rule_based_fallback",
                "timestamp": current_time,
                "confidence": 0.6
            }
        else:
            return {
                "action": "wait",
                "wait_time": 120,
                "reason": "no_ready_clusters",
                "timestamp": current_time,
                "confidence": 0.5
            }
            
    async def learn_from_feedback(self, 
                                decision: Dict[str, Any],
                                outcome_metrics: Dict[str, float],
                                user_feedback: Optional[Dict[str, float]] = None):
        """Learn from decision outcomes"""
        try:
            # Store training data
            training_sample = {
                "decision": decision,
                "outcome": outcome_metrics,
                "user_feedback": user_feedback or {},
                "timestamp": datetime.utcnow()
            }
            
            self.training_data.append(training_sample)
            
            # Update user preferences
            if user_feedback:
                for topic, rating in user_feedback.items():
                    # Exponential moving average
                    alpha = 0.1
                    current_pref = self.user_preferences[topic]
                    self.user_preferences[topic] = alpha * rating + (1 - alpha) * current_pref
                    
            # Trigger training if enough data accumulated
            if len(self.training_data) >= self.config.RL_UPDATE_FREQUENCY:
                await self._update_model()
                
        except Exception as e:
            logger.error(f"Error learning from feedback: {e}")
            
    async def _update_model(self):
        """Update RL model with accumulated training data"""
        try:
            if not self.model or not self.env:
                return
                
            logger.info("Updating RL model with recent experiences...")
            
            # Train model with recent experiences
            # This is simplified - in practice, you'd need to properly format
            # the training data for the RL algorithm
            
            self.model.learn(total_timesteps=len(self.training_data))
            
            # Save updated model
            model_path = self.config.RL_MODEL_PATH
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.model.save(model_path)
            
            # Clear training data
            self.training_data.clear()
            
            logger.info("RL model updated and saved")
            
        except Exception as e:
            logger.error(f"Error updating RL model: {e}")
            
    async def train_agent(self, num_episodes: int = None):
        """Train the RL agent"""
        if num_episodes is None:
            num_episodes = self.config.RL_TRAINING_EPISODES
            
        try:
            logger.info(f"Training RL agent for {num_episodes} episodes...")
            
            if not self.model or not self.env:
                await self.initialize()
                
            # Train the model
            self.model.learn(total_timesteps=num_episodes * 100)  # 100 steps per episode
            
            # Save trained model
            model_path = self.config.RL_MODEL_PATH
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.model.save(model_path)
            
            logger.info("RL agent training completed")
            
        except Exception as e:
            logger.error(f"Error training RL agent: {e}")
            
    async def get_agent_statistics(self) -> Dict[str, Any]:
        """Get RL agent statistics"""
        recent_decisions = [
            d for d in self.decision_history
            if (datetime.utcnow() - d["timestamp"]).total_seconds() < 3600
        ]
        
        action_counts = defaultdict(int)
        for decision in recent_decisions:
            action_counts[decision["decision"]["action"]] += 1
            
        return {
            "total_decisions": len(self.decision_history),
            "recent_decisions": len(recent_decisions),
            "action_distribution": dict(action_counts),
            "training_samples": len(self.training_data),
            "user_preferences": dict(self.user_preferences),
            "model_loaded": self.model is not None,
            "average_confidence": np.mean([
                d["decision"].get("confidence", 0.5) 
                for d in recent_decisions
            ]) if recent_decisions else 0.5
        }
        
    async def cleanup(self):
        """Cleanup RL agent resources"""
        # Save final model state
        if self.model:
            try:
                model_path = self.config.RL_MODEL_PATH
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                self.model.save(model_path)
            except Exception as e:
                logger.error(f"Error saving model during cleanup: {e}")
                
        # Save user preferences
        try:
            prefs_path = f"{self.config.RL_MODEL_PATH}_preferences.pkl"
            with open(prefs_path, 'wb') as f:
                pickle.dump(dict(self.user_preferences), f)
        except Exception as e:
            logger.error(f"Error saving user preferences: {e}")
            
        self.training_data.clear()
        self.decision_history.clear()
        
        logger.info("RL agent cleanup completed")
