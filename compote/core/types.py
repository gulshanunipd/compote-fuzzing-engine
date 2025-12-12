"""
Core data types and structures for COMPOTE fuzzing engine.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Tuple
from enum import Enum
import time
import numpy as np


class MessageType(Enum):
    """Consensus message types"""
    PROPOSE = "propose"
    PREVOTE = "prevote" 
    PRECOMMIT = "precommit"
    COMMIT = "commit"
    ROUND_CHANGE = "round_change"
    NEW_VIEW = "new_view"
    PREPARE = "prepare"
    VIEW_CHANGE = "view_change"
    OTHER = "other"


class NodeRole(Enum):
    """Node roles in consensus"""
    LEADER = "leader"
    VALIDATOR = "validator" 
    OBSERVER = "observer"
    PROPOSER = "proposer"
    UNKNOWN = "unknown"


@dataclass
class RawMessage:
    """Raw consensus message as received from network"""
    data: bytes
    timestamp: float
    source_id: str
    message_id: str = field(default_factory=lambda: str(time.time_ns()))
    size: int = field(init=False)
    
    def __post_init__(self):
        self.size = len(self.data)


@dataclass 
class ParsedMessage:
    """Parsed consensus message with extracted fields"""
    message_id: str
    message_type: MessageType
    round_number: int
    view_number: int
    block_height: int
    sender_id: str
    sender_role: NodeRole
    timestamp: float
    payload_hash: str
    signature: Optional[str] = None
    additional_fields: Dict[str, Any] = field(default_factory=dict)
    
    # Computed fields
    latency: float = 0.0
    round_deviation: float = 0.0
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParsedMessage':
        """Create ParsedMessage from dictionary"""
        return cls(**data)


@dataclass
class MessageFeatures:
    """Numerical feature vector for clustering and priority calculation"""
    message_id: str
    features: np.ndarray
    feature_names: List[str]
    similarity_scores: Dict[str, float] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.features)
    
    def get_feature(self, name: str) -> float:
        """Get feature value by name"""
        if name in self.feature_names:
            idx = self.feature_names.index(name)
            return self.features[idx]
        raise KeyError(f"Feature '{name}' not found")


@dataclass
class MessageCluster:
    """A cluster of similar messages (context pool)"""
    cluster_id: int
    message_ids: List[str]
    centroid: np.ndarray
    feature_names: List[str]
    priority_scores: Dict[str, float] = field(default_factory=dict)
    
    def add_message(self, message_id: str, priority: float = 0.0):
        """Add message to cluster"""
        if message_id not in self.message_ids:
            self.message_ids.append(message_id)
            self.priority_scores[message_id] = priority
    
    def get_highest_priority_message(self) -> Optional[str]:
        """Get message ID with highest priority"""
        if not self.priority_scores:
            return None
        return max(self.priority_scores.items(), key=lambda x: x[1])[0]


@dataclass 
class PriorityScore:
    """Priority score components for a message"""
    message_id: str
    similarity_score: float
    fault_score: float
    coverage_score: float
    total_score: float
    weights: Dict[str, float] = field(default_factory=lambda: {
        'alpha': 0.3,  # similarity weight
        'beta': 0.4,   # fault weight  
        'gamma': 0.3   # coverage weight
    })
    
    @classmethod
    def calculate(cls, message_id: str, similarity: float, fault: float, 
                  coverage: float, weights: Optional[Dict[str, float]] = None) -> 'PriorityScore':
        """Calculate total priority score"""
        w = weights or {'alpha': 0.3, 'beta': 0.4, 'gamma': 0.3}
        total = w['alpha'] * similarity + w['beta'] * fault + w['gamma'] * coverage
        return cls(message_id, similarity, fault, coverage, total, w)


@dataclass
class ExecutionResult:
    """Result of executing a mutated message"""
    message_id: str
    execution_time: float
    success: bool
    state_changes: List[str]
    coverage_metrics: Dict[str, float]
    fault_detected: bool = False
    error_message: Optional[str] = None
    new_paths_covered: int = 0
    
    
@dataclass
class HistoricalRecord:
    """Historical execution record for feedback"""
    message_id: str
    execution_results: List[ExecutionResult]
    fault_count: int = 0
    total_executions: int = 0
    last_execution: float = 0.0
    
    def update(self, result: ExecutionResult):
        """Update historical record with new execution"""
        self.execution_results.append(result)
        self.total_executions += 1
        self.last_execution = time.time()
        if result.fault_detected:
            self.fault_count += 1
    
    @property
    def fault_rate(self) -> float:
        """Calculate fault rate"""
        return self.fault_count / max(1, self.total_executions)


@dataclass
class FuzzingState:
    """Current state of the fuzzing process"""
    iteration: int
    total_messages: int
    clusters_count: int
    messages_executed: int
    faults_found: int
    coverage_percentage: float
    current_priorities: Dict[str, float] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    
    @property
    def runtime(self) -> float:
        """Get current runtime in seconds"""
        return time.time() - self.start_time