"""
Algorithm 2: Feature Extraction

Purpose: Convert parsed messages into numerical vectors for clustering.
Complexity: O(M·f) for M messages and f features.

Extracts message-level features such as:
- Message type indicators
- Round deviation  
- Sender role weight
- Message latency
- Temporal features
- Structural features
"""

import numpy as np
import time
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from ..core.types import ParsedMessage, MessageFeatures, MessageType, NodeRole


class FeatureExtractor:
    """
    Extracts numerical features from parsed consensus messages.
    
    Features are categorized into:
    1. Categorical features (message type, sender role)
    2. Numerical features (round, view, height, timestamp)
    3. Derived features (latency, deviation, weights)
    4. Structural features (payload size, signature presence)
    """
    
    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.feature_names = []
        self.feature_stats = {}  # For normalization
        self.global_stats = {
            'round_mean': 0.0,
            'round_std': 1.0,
            'view_mean': 0.0, 
            'view_std': 1.0,
            'height_mean': 0.0,
            'height_std': 1.0,
            'latency_mean': 0.0,
            'latency_std': 1.0
        }
        self._setup_feature_names()
    
    def _setup_feature_names(self):
        """Initialize feature names in consistent order"""
        self.feature_names = [
            # Categorical features (one-hot encoded)
            'msg_type_propose', 'msg_type_prevote', 'msg_type_precommit', 
            'msg_type_commit', 'msg_type_round_change', 'msg_type_new_view',
            'msg_type_prepare', 'msg_type_view_change', 'msg_type_other',
            
            # Node role features
            'role_leader', 'role_validator', 'role_observer', 'role_proposer', 'role_unknown',
            
            # Numerical features
            'round_number', 'view_number', 'block_height',
            
            # Temporal features
            'timestamp_normalized', 'message_latency', 'round_deviation',
            
            # Structural features  
            'payload_hash_entropy', 'signature_present', 'additional_fields_count',
            
            # Derived features
            'sender_role_weight', 'message_complexity', 'temporal_distance'
        ]
    
    def extract_features(self, messages: List[ParsedMessage]) -> List[MessageFeatures]:
        """
        Extract features from a list of parsed messages.
        
        Args:
            messages: List of parsed consensus messages
            
        Returns:
            List of MessageFeatures with numerical vectors
        """
        if not messages:
            return []
        
        # Update global statistics for normalization
        self._update_global_stats(messages)
        
        feature_list = []
        for msg in messages:
            features = self._extract_single_message_features(msg, messages)
            feature_list.append(features)
        
        return feature_list
    
    def _extract_single_message_features(self, message: ParsedMessage, 
                                       all_messages: List[ParsedMessage]) -> MessageFeatures:
        """Extract features for a single message"""
        
        # Initialize feature vector
        feature_vector = np.zeros(len(self.feature_names))
        
        # 1. Categorical features - Message type (one-hot encoding)
        msg_type_idx = self._get_message_type_index(message.message_type)
        if msg_type_idx >= 0:
            feature_vector[msg_type_idx] = 1.0
        
        # 2. Categorical features - Node role (one-hot encoding)  
        role_idx = self._get_role_index(message.sender_role)
        if role_idx >= 0:
            feature_vector[role_idx] = 1.0
        
        # 3. Numerical features
        feature_vector[self.feature_names.index('round_number')] = message.round_number
        feature_vector[self.feature_names.index('view_number')] = message.view_number
        feature_vector[self.feature_names.index('block_height')] = message.block_height
        
        # 4. Temporal features
        feature_vector[self.feature_names.index('timestamp_normalized')] = message.timestamp
        feature_vector[self.feature_names.index('message_latency')] = self._calculate_latency(message, all_messages)
        feature_vector[self.feature_names.index('round_deviation')] = self._calculate_round_deviation(message, all_messages)
        
        # 5. Structural features
        feature_vector[self.feature_names.index('payload_hash_entropy')] = self._calculate_hash_entropy(message.payload_hash)
        feature_vector[self.feature_names.index('signature_present')] = 1.0 if message.signature else 0.0
        feature_vector[self.feature_names.index('additional_fields_count')] = len(message.additional_fields)
        
        # 6. Derived features
        feature_vector[self.feature_names.index('sender_role_weight')] = self._get_role_weight(message.sender_role)
        feature_vector[self.feature_names.index('message_complexity')] = self._calculate_complexity(message)
        feature_vector[self.feature_names.index('temporal_distance')] = self._calculate_temporal_distance(message, all_messages)
        
        # 7. Normalize features if required
        if self.normalize:
            feature_vector = self._normalize_features(feature_vector)
        
        return MessageFeatures(
            message_id=message.message_id,
            features=feature_vector,
            feature_names=self.feature_names.copy()
        )
    
    def _get_message_type_index(self, msg_type: MessageType) -> int:
        """Get index for message type one-hot encoding"""
        type_mapping = {
            MessageType.PROPOSE: 0,
            MessageType.PREVOTE: 1, 
            MessageType.PRECOMMIT: 2,
            MessageType.COMMIT: 3,
            MessageType.ROUND_CHANGE: 4,
            MessageType.NEW_VIEW: 5,
            MessageType.PREPARE: 6,
            MessageType.VIEW_CHANGE: 7,
            MessageType.OTHER: 8
        }
        return type_mapping.get(msg_type, 8)  # Default to 'other'
    
    def _get_role_index(self, role: NodeRole) -> int:
        """Get index for node role one-hot encoding"""
        role_mapping = {
            NodeRole.LEADER: 9,
            NodeRole.VALIDATOR: 10,
            NodeRole.OBSERVER: 11, 
            NodeRole.PROPOSER: 12,
            NodeRole.UNKNOWN: 13
        }
        return role_mapping.get(role, 13)  # Default to 'unknown'
    
    def _calculate_latency(self, message: ParsedMessage, all_messages: List[ParsedMessage]) -> float:
        """Calculate message latency relative to round start"""
        # Find earliest message in same round
        same_round_messages = [m for m in all_messages if m.round_number == message.round_number]
        if not same_round_messages:
            return 0.0
        
        earliest_timestamp = min(m.timestamp for m in same_round_messages)
        return message.timestamp - earliest_timestamp
    
    def _calculate_round_deviation(self, message: ParsedMessage, all_messages: List[ParsedMessage]) -> float:
        """Calculate how much this message's round deviates from average"""
        if not all_messages:
            return 0.0
        
        round_numbers = [m.round_number for m in all_messages]
        mean_round = np.mean(round_numbers)
        return abs(message.round_number - mean_round)
    
    def _calculate_hash_entropy(self, payload_hash: str) -> float:
        """Calculate entropy of payload hash (measure of randomness)"""
        if not payload_hash:
            return 0.0
        
        # Calculate byte frequency entropy
        byte_freq = defaultdict(int)
        for char in payload_hash:
            byte_freq[char] += 1
        
        length = len(payload_hash)
        entropy = 0.0
        for count in byte_freq.values():
            prob = count / length
            if prob > 0:
                entropy -= prob * np.log2(prob)
        
        return entropy
    
    def _get_role_weight(self, role: NodeRole) -> float:
        """Get numerical weight for node role"""
        weights = {
            NodeRole.LEADER: 1.0,
            NodeRole.PROPOSER: 0.9,
            NodeRole.VALIDATOR: 0.7,
            NodeRole.OBSERVER: 0.3,
            NodeRole.UNKNOWN: 0.1
        }
        return weights.get(role, 0.1)
    
    def _calculate_complexity(self, message: ParsedMessage) -> float:
        """Calculate message complexity score"""
        complexity = 0.0
        
        # Base complexity from message type
        type_complexity = {
            MessageType.PROPOSE: 1.0,
            MessageType.COMMIT: 0.8,
            MessageType.PRECOMMIT: 0.6,
            MessageType.PREVOTE: 0.4,
            MessageType.ROUND_CHANGE: 0.9,
            MessageType.NEW_VIEW: 0.9,
            MessageType.PREPARE: 0.5,
            MessageType.VIEW_CHANGE: 0.8,
            MessageType.OTHER: 0.2
        }
        complexity += type_complexity.get(message.message_type, 0.2)
        
        # Additional complexity from extra fields
        complexity += len(message.additional_fields) * 0.1
        
        # Signature adds complexity
        if message.signature:
            complexity += 0.2
        
        return min(complexity, 2.0)  # Cap at 2.0
    
    def _calculate_temporal_distance(self, message: ParsedMessage, all_messages: List[ParsedMessage]) -> float:
        """Calculate temporal distance from most recent message"""
        if not all_messages:
            return 0.0
        
        latest_timestamp = max(m.timestamp for m in all_messages)
        return latest_timestamp - message.timestamp
    
    def _update_global_stats(self, messages: List[ParsedMessage]):
        """Update global statistics for normalization"""
        if not messages:
            return
        
        rounds = [m.round_number for m in messages]
        views = [m.view_number for m in messages]
        heights = [m.block_height for m in messages]
        timestamps = [m.timestamp for m in messages]
        
        # Calculate statistics
        self.global_stats['round_mean'] = np.mean(rounds)
        self.global_stats['round_std'] = np.std(rounds) or 1.0
        
        self.global_stats['view_mean'] = np.mean(views)
        self.global_stats['view_std'] = np.std(views) or 1.0
        
        self.global_stats['height_mean'] = np.mean(heights)
        self.global_stats['height_std'] = np.std(heights) or 1.0
        
        # Calculate latencies for normalization
        latencies = []
        for msg in messages:
            latency = self._calculate_latency(msg, messages)
            latencies.append(latency)
        
        self.global_stats['latency_mean'] = np.mean(latencies)
        self.global_stats['latency_std'] = np.std(latencies) or 1.0
    
    def _normalize_features(self, feature_vector: np.ndarray) -> np.ndarray:
        """Normalize numerical features using z-score normalization"""
        normalized = feature_vector.copy()
        
        # Normalize specific numerical features
        round_idx = self.feature_names.index('round_number')
        normalized[round_idx] = (feature_vector[round_idx] - self.global_stats['round_mean']) / self.global_stats['round_std']
        
        view_idx = self.feature_names.index('view_number')
        normalized[view_idx] = (feature_vector[view_idx] - self.global_stats['view_mean']) / self.global_stats['view_std']
        
        height_idx = self.feature_names.index('block_height')
        normalized[height_idx] = (feature_vector[height_idx] - self.global_stats['height_mean']) / self.global_stats['height_std']
        
        latency_idx = self.feature_names.index('message_latency')
        normalized[latency_idx] = (feature_vector[latency_idx] - self.global_stats['latency_mean']) / self.global_stats['latency_std']
        
        # Normalize timestamp to [0, 1] range
        timestamp_idx = self.feature_names.index('timestamp_normalized')
        if hasattr(self, '_min_timestamp') and hasattr(self, '_max_timestamp'):
            timestamp_range = self._max_timestamp - self._min_timestamp
            if timestamp_range > 0:
                normalized[timestamp_idx] = (feature_vector[timestamp_idx] - self._min_timestamp) / timestamp_range
        
        return normalized
    
    def compute_similarity(self, features1: MessageFeatures, features2: MessageFeatures) -> float:
        """Compute similarity between two feature vectors using cosine similarity"""
        # Cosine similarity
        dot_product = np.dot(features1.features, features2.features)
        norm1 = np.linalg.norm(features1.features)
        norm2 = np.linalg.norm(features2.features)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return max(0.0, similarity)  # Ensure non-negative
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get relative importance of different feature categories"""
        importance = {}
        
        # Categorical features
        categorical_count = 14  # 9 message types + 5 roles
        importance['categorical'] = categorical_count / len(self.feature_names)
        
        # Numerical features  
        numerical_count = 3  # round, view, height
        importance['numerical'] = numerical_count / len(self.feature_names)
        
        # Temporal features
        temporal_count = 3  # timestamp, latency, deviation
        importance['temporal'] = temporal_count / len(self.feature_names)
        
        # Structural features
        structural_count = 3  # entropy, signature, fields
        importance['structural'] = structural_count / len(self.feature_names)
        
        # Derived features
        derived_count = 3  # role weight, complexity, temporal distance
        importance['derived'] = derived_count / len(self.feature_names)
        
        return importance
    
    def get_stats(self) -> Dict[str, Any]:
        """Get feature extraction statistics"""
        return {
            'total_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'normalization_enabled': self.normalize,
            'global_stats': self.global_stats,
            'feature_importance': self.get_feature_importance()
        }