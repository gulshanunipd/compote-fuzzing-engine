"""
Algorithm 4: Priority Calculation

Purpose: Assign priority score P to each message within its cluster.
Complexity: O(n·|T|) but with pruning to maintain efficiency.

Priority = α·similarity + β·fault + γ·coverage

Includes threshold-based pruning:
- Fault impact threshold filters messages with low bug-inducing history
- Similarity relevance threshold ensures relevance to current targets  
- Time expiration threshold removes stale messages
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict, deque
import logging
from ..core.types import (
    MessageFeatures, MessageCluster, PriorityScore, HistoricalRecord, 
    ExecutionResult, ParsedMessage
)


class PriorityCalculator:
    """
    Calculates priority scores for messages within clusters using weighted scoring.
    
    Combines three main factors:
    1. Similarity scores to other messages in cluster
    2. Historical fault likelihood from execution logs
    3. Code coverage contribution from past mutations
    
    Applies threshold-based pruning for efficiency.
    """
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.4, gamma: float = 0.3,
                 fault_threshold: float = 0.1, similarity_threshold: float = 0.2,
                 time_threshold: float = 3600.0, max_history_size: int = 1000):
        """
        Initialize priority calculator.
        
        Args:
            alpha: Weight for similarity score
            beta: Weight for fault score  
            gamma: Weight for coverage score
            fault_threshold: Minimum fault impact to consider
            similarity_threshold: Minimum similarity relevance
            time_threshold: Time expiration threshold (seconds)
            max_history_size: Maximum historical records to keep
        """
        # Scoring weights
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Pruning thresholds
        self.fault_threshold = fault_threshold
        self.similarity_threshold = similarity_threshold
        self.time_threshold = time_threshold
        self.max_history_size = max_history_size
        
        # Historical data
        self.historical_records: Dict[str, HistoricalRecord] = {}
        self.coverage_data: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Current cluster context
        self.current_clusters: Dict[int, MessageCluster] = {}
        self.message_features: Dict[str, MessageFeatures] = {}
        
        # Priority cache for efficiency
        self.priority_cache: Dict[str, PriorityScore] = {}
        self.cache_timestamp = time.time()
        self.cache_expiry = 300.0  # 5 minutes
        
        # Statistics
        self.calculation_stats = {
            'total_calculations': 0,
            'cache_hits': 0,
            'pruned_messages': 0,
            'expired_records': 0
        }
        
        self.logger = logging.getLogger(__name__)
    
    def calculate_priorities(self, clusters: Dict[int, MessageCluster], 
                           message_features: Dict[str, MessageFeatures],
                           force_recalculate: bool = False) -> Dict[str, PriorityScore]:
        """
        Calculate priority scores for all messages in clusters.
        
        Args:
            clusters: Dictionary of message clusters
            message_features: Features for each message
            force_recalculate: Force recalculation even if cached
            
        Returns:
            Dictionary mapping message_id to PriorityScore
        """
        self.current_clusters = clusters
        self.message_features = message_features
        
        # Check cache validity
        if not force_recalculate and self._is_cache_valid():
            self.calculation_stats['cache_hits'] += 1
            return self.priority_cache
        
        # Clear expired cache
        if time.time() - self.cache_timestamp > self.cache_expiry:
            self.priority_cache.clear()
            self.cache_timestamp = time.time()
        
        # Prune old historical records
        self._prune_historical_records()
        
        priority_scores = {}
        
        for cluster_id, cluster in clusters.items():
            if cluster_id < 0:  # Skip noise clusters
                continue
            
            # Calculate priorities for messages in this cluster
            cluster_priorities = self._calculate_cluster_priorities(cluster)
            priority_scores.update(cluster_priorities)
        
        # Update cache
        self.priority_cache = priority_scores
        self.cache_timestamp = time.time()
        
        self.calculation_stats['total_calculations'] += 1
        return priority_scores
    
    def _calculate_cluster_priorities(self, cluster: MessageCluster) -> Dict[str, PriorityScore]:
        """Calculate priorities for messages within a single cluster"""
        cluster_priorities = {}
        
        for message_id in cluster.message_ids:
            if message_id not in self.message_features:
                self.logger.warning(f"No features found for message {message_id}")
                continue
            
            # Apply pruning thresholds
            if not self._should_calculate_priority(message_id):
                self.calculation_stats['pruned_messages'] += 1
                continue
            
            # Calculate individual score components
            similarity_score = self._calculate_similarity_score(message_id, cluster)
            fault_score = self._calculate_fault_score(message_id)
            coverage_score = self._calculate_coverage_score(message_id)
            
            # Create priority score
            priority = PriorityScore.calculate(
                message_id=message_id,
                similarity=similarity_score,
                fault=fault_score,
                coverage=coverage_score,
                weights={'alpha': self.alpha, 'beta': self.beta, 'gamma': self.gamma}
            )
            
            cluster_priorities[message_id] = priority
            
            # Update cluster priority mapping
            cluster.priority_scores[message_id] = priority.total_score
        
        return cluster_priorities
    
    def _should_calculate_priority(self, message_id: str) -> bool:
        """Apply threshold-based pruning to determine if priority should be calculated"""
        
        # 1. Time expiration threshold
        if message_id in self.historical_records:
            record = self.historical_records[message_id]
            time_elapsed = time.time() - record.last_execution
            if time_elapsed > self.time_threshold:
                return False
        
        # 2. Fault impact threshold
        fault_rate = self._get_fault_rate(message_id)
        if fault_rate < self.fault_threshold:
            return False
        
        # 3. Similarity relevance threshold (check against cluster centroid)
        similarity_to_cluster = self._calculate_cluster_relevance(message_id)
        if similarity_to_cluster < self.similarity_threshold:
            return False
        
        return True
    
    def _calculate_similarity_score(self, message_id: str, cluster: MessageCluster) -> float:
        """Calculate similarity score within cluster"""
        if message_id not in self.message_features:
            return 0.0
        
        target_features = self.message_features[message_id]
        similarities = []
        
        # Calculate similarity to other messages in cluster
        for other_id in cluster.message_ids:
            if other_id == message_id or other_id not in self.message_features:
                continue
            
            other_features = self.message_features[other_id]
            similarity = self._cosine_similarity(target_features.features, other_features.features)
            similarities.append(similarity)
        
        # Calculate similarity to cluster centroid
        centroid_similarity = self._cosine_similarity(target_features.features, cluster.centroid)
        similarities.append(centroid_similarity)
        
        # Return average similarity
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_fault_score(self, message_id: str) -> float:
        """Calculate historical fault likelihood score"""
        if message_id not in self.historical_records:
            return 0.0
        
        record = self.historical_records[message_id]
        
        # Base fault rate
        fault_rate = record.fault_rate
        
        # Weight by recency (more recent faults get higher score)
        time_decay = self._calculate_time_decay(record.last_execution)
        
        # Weight by execution frequency (more executions = more reliable data)
        execution_weight = min(1.0, record.total_executions / 10.0)
        
        return fault_rate * time_decay * execution_weight
    
    def _calculate_coverage_score(self, message_id: str) -> float:
        """Calculate code coverage contribution score"""
        if message_id not in self.coverage_data:
            return 0.0
        
        coverage_info = self.coverage_data[message_id]
        
        # Aggregate different coverage metrics
        line_coverage = coverage_info.get('line_coverage', 0.0)
        branch_coverage = coverage_info.get('branch_coverage', 0.0)
        function_coverage = coverage_info.get('function_coverage', 0.0)
        new_paths = coverage_info.get('new_paths_discovered', 0.0)
        
        # Weighted combination of coverage metrics
        coverage_score = (
            0.3 * line_coverage +
            0.3 * branch_coverage +
            0.2 * function_coverage +
            0.2 * min(1.0, new_paths / 10.0)  # Normalize new paths
        )
        
        return coverage_score
    
    def _calculate_cluster_relevance(self, message_id: str) -> float:
        """Calculate relevance of message to its cluster"""
        if message_id not in self.message_features:
            return 0.0
        
        message_features = self.message_features[message_id]
        
        # Find the cluster containing this message
        for cluster in self.current_clusters.values():
            if message_id in cluster.message_ids:
                return self._cosine_similarity(message_features.features, cluster.centroid)
        
        return 0.0
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        if len(vec1) != len(vec2):
            return 0.0
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def _calculate_time_decay(self, last_execution: float) -> float:
        """Calculate time decay factor (more recent = higher score)"""
        if last_execution == 0:
            return 0.0
        
        time_elapsed = time.time() - last_execution
        # Exponential decay with half-life of 1 hour
        half_life = 3600.0
        return np.exp(-time_elapsed * np.log(2) / half_life)
    
    def _get_fault_rate(self, message_id: str) -> float:
        """Get fault rate for a message"""
        if message_id not in self.historical_records:
            return 0.0
        return self.historical_records[message_id].fault_rate
    
    def _prune_historical_records(self):
        """Remove old and irrelevant historical records"""
        current_time = time.time()
        expired_ids = []
        
        for message_id, record in self.historical_records.items():
            # Remove records older than time threshold
            if current_time - record.last_execution > self.time_threshold:
                expired_ids.append(message_id)
        
        for message_id in expired_ids:
            del self.historical_records[message_id]
            self.calculation_stats['expired_records'] += 1
        
        # Limit total number of records
        if len(self.historical_records) > self.max_history_size:
            # Remove oldest records
            sorted_records = sorted(self.historical_records.items(), 
                                  key=lambda x: x[1].last_execution)
            excess_count = len(self.historical_records) - self.max_history_size
            
            for i in range(excess_count):
                message_id = sorted_records[i][0]
                del self.historical_records[message_id]
    
    def _is_cache_valid(self) -> bool:
        """Check if priority cache is still valid"""
        return (time.time() - self.cache_timestamp) < self.cache_expiry
    
    def update_execution_result(self, message_id: str, result: ExecutionResult):
        """Update historical records with new execution result"""
        if message_id not in self.historical_records:
            self.historical_records[message_id] = HistoricalRecord(
                message_id=message_id,
                execution_results=[]
            )
        
        # Update historical record
        self.historical_records[message_id].update(result)
        
        # Update coverage data
        if result.coverage_metrics:
            self.coverage_data[message_id].update(result.coverage_metrics)
            self.coverage_data[message_id]['new_paths_discovered'] = result.new_paths_covered
        
        # Invalidate cache
        self.priority_cache.clear()
    
    def get_top_priority_messages(self, cluster_id: int, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get top-k highest priority messages from a cluster"""
        if cluster_id not in self.current_clusters:
            return []
        
        cluster = self.current_clusters[cluster_id]
        
        # Get priority scores for messages in cluster
        message_priorities = []
        for message_id in cluster.message_ids:
            if message_id in self.priority_cache:
                priority = self.priority_cache[message_id].total_score
                message_priorities.append((message_id, priority))
        
        # Sort by priority and return top-k
        message_priorities.sort(key=lambda x: x[1], reverse=True)
        return message_priorities[:top_k]
    
    def update_weights(self, alpha: float, beta: float, gamma: float):
        """Update priority calculation weights"""
        # Normalize weights to sum to 1
        total = alpha + beta + gamma
        if total > 0:
            self.alpha = alpha / total
            self.beta = beta / total  
            self.gamma = gamma / total
        
        # Invalidate cache
        self.priority_cache.clear()
    
    def get_priority_distribution(self) -> Dict[str, any]:
        """Get statistics about priority score distribution"""
        if not self.priority_cache:
            return {}
        
        scores = [p.total_score for p in self.priority_cache.values()]
        similarity_scores = [p.similarity_score for p in self.priority_cache.values()]
        fault_scores = [p.fault_score for p in self.priority_cache.values()]
        coverage_scores = [p.coverage_score for p in self.priority_cache.values()]
        
        return {
            'total_scores': {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'median': np.median(scores)
            },
            'component_scores': {
                'similarity': {
                    'mean': np.mean(similarity_scores),
                    'contribution': self.alpha
                },
                'fault': {
                    'mean': np.mean(fault_scores),
                    'contribution': self.beta
                },
                'coverage': {
                    'mean': np.mean(coverage_scores),
                    'contribution': self.gamma
                }
            },
            'thresholds': {
                'fault_threshold': self.fault_threshold,
                'similarity_threshold': self.similarity_threshold,
                'time_threshold': self.time_threshold
            },
            'statistics': self.calculation_stats
        }
    
    def export_priorities(self, file_path: str):
        """Export current priorities to file"""
        import json
        
        export_data = {
            'timestamp': time.time(),
            'weights': {'alpha': self.alpha, 'beta': self.beta, 'gamma': self.gamma},
            'thresholds': {
                'fault_threshold': self.fault_threshold,
                'similarity_threshold': self.similarity_threshold,
                'time_threshold': self.time_threshold
            },
            'priorities': {
                msg_id: {
                    'total_score': score.total_score,
                    'similarity_score': score.similarity_score,
                    'fault_score': score.fault_score,
                    'coverage_score': score.coverage_score
                }
                for msg_id, score in self.priority_cache.items()
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Exported priorities to {file_path}")
    
    def get_calculation_stats(self) -> Dict[str, any]:
        """Get detailed statistics about priority calculations"""
        return {
            'weights': {'alpha': self.alpha, 'beta': self.beta, 'gamma': self.gamma},
            'thresholds': {
                'fault_threshold': self.fault_threshold,
                'similarity_threshold': self.similarity_threshold,
                'time_threshold': self.time_threshold
            },
            'cache_info': {
                'cache_size': len(self.priority_cache),
                'cache_age': time.time() - self.cache_timestamp,
                'cache_expiry': self.cache_expiry
            },
            'historical_data': {
                'records_count': len(self.historical_records),
                'coverage_entries': len(self.coverage_data)
            },
            'statistics': self.calculation_stats
        }