"""
Algorithm 5: State Analysis & Priority Update

Purpose: Drive fuzzing iterations and update the seed pool.

Main loop:
1. Select highest-priority message from cluster
2. Mutate fields (timestamps, round numbers, etc.)  
3. Execute mutated message on blockchain network
4. Record state transitions and coverage metrics
5. Re-compute similarity, fault scores, and coverage
6. Update priority queue

This creates the feedback loop that continuously refines the seed pool.
"""

import time
import random
import hashlib
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import defaultdict
from dataclasses import dataclass, field
import logging
from ..core.types import (
    ParsedMessage, MessageCluster, ExecutionResult, FuzzingState, 
    MessageType, NodeRole, PriorityScore
)


@dataclass
class MutationStrategy:
    """Configuration for message mutation strategies"""
    timestamp_variance: float = 0.1  # ±10% timestamp mutation
    round_variance: int = 2          # ±2 rounds
    view_variance: int = 1           # ±1 view
    field_corruption_rate: float = 0.05  # 5% field corruption chance
    payload_mutation_rate: float = 0.1   # 10% payload mutation
    enabled_mutations: List[str] = field(default_factory=lambda: [
        'timestamp', 'round', 'view', 'payload', 'sender'
    ])


@dataclass 
class ExecutionEnvironment:
    """Configuration for test execution environment"""
    network_nodes: List[str] = field(default_factory=list)
    execution_timeout: float = 30.0
    max_retries: int = 3
    enable_coverage_tracking: bool = True
    enable_state_monitoring: bool = True
    simulation_mode: bool = False  # For testing without real network


class StateAnalyzer:
    """
    Implements the core fuzzing loop with state analysis and priority updates.
    
    Manages the iterative process of:
    - Selecting high-priority messages
    - Applying mutations
    - Executing on target system
    - Analyzing results and updating priorities
    """
    
    def __init__(self, mutation_strategy: Optional[MutationStrategy] = None,
                 execution_env: Optional[ExecutionEnvironment] = None):
        """
        Initialize state analyzer.
        
        Args:
            mutation_strategy: Configuration for message mutations
            execution_env: Configuration for execution environment
        """
        self.mutation_strategy = mutation_strategy or MutationStrategy()
        self.execution_env = execution_env or ExecutionEnvironment()
        
        # Current fuzzing state
        self.fuzzing_state = FuzzingState(
            iteration=0,
            total_messages=0,
            clusters_count=0,
            messages_executed=0,
            faults_found=0,
            coverage_percentage=0.0
        )
        
        # Execution history and metrics
        self.execution_history: List[ExecutionResult] = []
        self.state_transitions: Dict[str, List[str]] = {}
        self.coverage_tracker = CoverageTracker()
        self.fault_detector = FaultDetector()
        
        # Mutation generators
        self.mutation_generators = {
            'timestamp': self._mutate_timestamp,
            'round': self._mutate_round_number,
            'view': self._mutate_view_number,
            'payload': self._mutate_payload,
            'sender': self._mutate_sender,
            'signature': self._mutate_signature
        }
        
        # Callback functions for external integration
        self.execution_callback: Optional[Callable] = None
        self.state_monitor_callback: Optional[Callable] = None
        self.fault_callback: Optional[Callable] = None
        
        self.logger = logging.getLogger(__name__)
    
    def run_fuzzing_iteration(self, clusters: Dict[int, MessageCluster], 
                            priority_calculator) -> ExecutionResult:
        """
        Run a single fuzzing iteration.
        
        Args:
            clusters: Current message clusters
            priority_calculator: Priority calculator instance
            
        Returns:
            ExecutionResult from this iteration
        """
        self.fuzzing_state.iteration += 1
        
        # Step 1: Select highest-priority message from clusters
        selected_message, cluster_id = self._select_highest_priority_message(clusters)
        if not selected_message:
            self.logger.warning("No message selected for mutation")
            return self._create_empty_result()
        
        # Step 2: Generate mutated message
        mutated_message = self._mutate_message(selected_message)
        
        # Step 3: Execute mutated message
        execution_result = self._execute_message(mutated_message)
        
        # Step 4: Analyze execution results
        self._analyze_execution_result(execution_result, mutated_message)
        
        # Step 5: Update state transitions
        self._update_state_transitions(mutated_message, execution_result)
        
        # Step 6: Update coverage metrics
        self._update_coverage_metrics(execution_result)
        
        # Step 7: Update priority calculator with new data
        priority_calculator.update_execution_result(mutated_message.message_id, execution_result)
        
        # Step 8: Update fuzzing state
        self._update_fuzzing_state(execution_result)
        
        # Store execution history
        self.execution_history.append(execution_result)
        
        return execution_result
    
    def _select_highest_priority_message(self, clusters: Dict[int, MessageCluster]) -> Tuple[Optional[ParsedMessage], int]:
        """Select the highest-priority message across all clusters"""
        best_message = None
        best_priority = -1.0
        best_cluster_id = -1
        
        for cluster_id, cluster in clusters.items():
            if cluster_id < 0:  # Skip noise clusters
                continue
            
            # Get highest priority message from this cluster
            highest_priority_id = cluster.get_highest_priority_message()
            if highest_priority_id and highest_priority_id in cluster.priority_scores:
                priority = cluster.priority_scores[highest_priority_id]
                
                if priority > best_priority:
                    best_priority = priority
                    best_cluster_id = cluster_id
                    # Create a basic ParsedMessage (in real implementation, would retrieve from storage)
                    best_message = self._create_message_from_id(highest_priority_id, cluster)
        
        return best_message, best_cluster_id
    
    def _create_message_from_id(self, message_id: str, cluster: MessageCluster) -> ParsedMessage:
        """Create a ParsedMessage object from message ID (placeholder implementation)"""
        # In real implementation, this would retrieve the actual message data
        # For now, create a basic message for demonstration
        return ParsedMessage(
            message_id=message_id,
            message_type=MessageType.PROPOSE,  # Would be retrieved from storage
            round_number=random.randint(1, 100),
            view_number=random.randint(0, 10),
            block_height=random.randint(1, 1000),
            sender_id=f"node_{random.randint(1, 10)}",
            sender_role=NodeRole.VALIDATOR,
            timestamp=time.time(),
            payload_hash=hashlib.sha256(message_id.encode()).hexdigest()
        )
    
    def _mutate_message(self, message: ParsedMessage) -> ParsedMessage:
        """Apply mutations to create a new test message"""
        mutated = ParsedMessage(
            message_id=f"{message.message_id}_mut_{int(time.time_ns())}",
            message_type=message.message_type,
            round_number=message.round_number,
            view_number=message.view_number,
            block_height=message.block_height,
            sender_id=message.sender_id,
            sender_role=message.sender_role,
            timestamp=message.timestamp,
            payload_hash=message.payload_hash,
            signature=message.signature,
            additional_fields=message.additional_fields.copy()
        )
        
        # Apply enabled mutations
        for mutation_type in self.mutation_strategy.enabled_mutations:
            if mutation_type in self.mutation_generators:
                if random.random() < self._get_mutation_probability(mutation_type):
                    self.mutation_generators[mutation_type](mutated)
        
        # Update payload hash after mutations
        mutated.payload_hash = self._calculate_payload_hash(mutated)
        
        return mutated
    
    def _get_mutation_probability(self, mutation_type: str) -> float:
        """Get probability of applying specific mutation type"""
        probabilities = {
            'timestamp': 0.8,
            'round': 0.6,
            'view': 0.4,
            'payload': self.mutation_strategy.payload_mutation_rate,
            'sender': 0.3,
            'signature': 0.2
        }
        return probabilities.get(mutation_type, 0.1)
    
    def _mutate_timestamp(self, message: ParsedMessage):
        """Mutate message timestamp"""
        variance = self.mutation_strategy.timestamp_variance
        current_time = time.time()
        
        # Add random variance to timestamp
        time_delta = random.uniform(-variance * current_time, variance * current_time)
        message.timestamp = max(0, message.timestamp + time_delta)
    
    def _mutate_round_number(self, message: ParsedMessage):
        """Mutate round number"""
        variance = self.mutation_strategy.round_variance
        delta = random.randint(-variance, variance)
        message.round_number = max(0, message.round_number + delta)
    
    def _mutate_view_number(self, message: ParsedMessage):
        """Mutate view number"""
        variance = self.mutation_strategy.view_variance
        delta = random.randint(-variance, variance)
        message.view_number = max(0, message.view_number + delta)
    
    def _mutate_payload(self, message: ParsedMessage):
        """Mutate payload-related fields"""
        # Randomly corrupt additional fields
        if message.additional_fields and random.random() < self.mutation_strategy.field_corruption_rate:
            field_name = random.choice(list(message.additional_fields.keys()))
            original_value = message.additional_fields[field_name]
            
            if isinstance(original_value, str):
                # String corruption
                if len(original_value) > 0:
                    pos = random.randint(0, len(original_value) - 1)
                    corrupted = list(original_value)
                    corrupted[pos] = chr(random.randint(32, 126))
                    message.additional_fields[field_name] = ''.join(corrupted)
            elif isinstance(original_value, int):
                # Integer corruption
                message.additional_fields[field_name] = original_value + random.randint(-100, 100)
    
    def _mutate_sender(self, message: ParsedMessage):
        """Mutate sender-related fields"""
        # Change sender ID
        if random.random() < 0.5:
            node_id = random.randint(1, 100)
            message.sender_id = f"node_{node_id}"
        
        # Change sender role
        if random.random() < 0.3:
            roles = [NodeRole.LEADER, NodeRole.VALIDATOR, NodeRole.OBSERVER]
            message.sender_role = random.choice(roles)
    
    def _mutate_signature(self, message: ParsedMessage):
        """Mutate or remove signature"""
        if random.random() < 0.5:
            # Remove signature
            message.signature = None
        else:
            # Corrupt signature
            if message.signature:
                corrupted = list(message.signature)
                if len(corrupted) > 0:
                    pos = random.randint(0, len(corrupted) - 1)
                    corrupted[pos] = chr(random.randint(32, 126))
                    message.signature = ''.join(corrupted)
    
    def _calculate_payload_hash(self, message: ParsedMessage) -> str:
        """Recalculate payload hash after mutations"""
        payload_data = f"{message.message_type.value}_{message.round_number}_{message.view_number}_{message.timestamp}"
        return hashlib.sha256(payload_data.encode()).hexdigest()
    
    def _execute_message(self, message: ParsedMessage) -> ExecutionResult:
        """Execute the mutated message on the target system"""
        start_time = time.time()
        
        try:
            if self.execution_env.simulation_mode:
                # Simulation mode for testing
                result = self._simulate_execution(message)
            else:
                # Real execution
                result = self._real_execution(message)
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"Execution failed for message {message.message_id}: {e}")
            return ExecutionResult(
                message_id=message.message_id,
                execution_time=time.time() - start_time,
                success=False,
                state_changes=[],
                coverage_metrics={},
                fault_detected=False,
                error_message=str(e)
            )
    
    def _simulate_execution(self, message: ParsedMessage) -> ExecutionResult:
        """Simulate message execution for testing purposes"""
        # Simulate random execution outcomes
        success = random.random() > 0.1  # 90% success rate
        fault_detected = random.random() < 0.05  # 5% fault detection rate
        
        state_changes = []
        if success:
            # Simulate state changes
            possible_states = ['round_advance', 'view_change', 'block_commit', 'leader_switch']
            num_changes = random.randint(0, 3)
            state_changes = random.sample(possible_states, min(num_changes, len(possible_states)))
        
        # Simulate coverage metrics
        coverage_metrics = {
            'line_coverage': random.uniform(0.1, 0.9),
            'branch_coverage': random.uniform(0.1, 0.8),
            'function_coverage': random.uniform(0.2, 0.7)
        }
        
        new_paths = random.randint(0, 5) if success else 0
        
        return ExecutionResult(
            message_id=message.message_id,
            execution_time=0.0,  # Will be set by caller
            success=success,
            state_changes=state_changes,
            coverage_metrics=coverage_metrics,
            fault_detected=fault_detected,
            new_paths_covered=new_paths
        )
    
    def _real_execution(self, message: ParsedMessage) -> ExecutionResult:
        """Execute message on real target system"""
        # Placeholder for real execution logic
        # In practice, this would:
        # 1. Connect to blockchain/consensus network
        # 2. Inject the mutated message
        # 3. Monitor system response
        # 4. Collect coverage and state data
        
        if self.execution_callback:
            return self.execution_callback(message)
        else:
            # Fallback to simulation
            return self._simulate_execution(message)
    
    def _analyze_execution_result(self, result: ExecutionResult, message: ParsedMessage):
        """Analyze execution result and detect faults"""
        # Update fault detection
        if result.fault_detected or not result.success:
            self.fault_detector.record_fault(message, result)
        
        # Analyze state transitions for anomalies
        if result.state_changes:
            self._analyze_state_transitions(message, result.state_changes)
        
        # Check for coverage improvements
        if result.new_paths_covered > 0:
            self.coverage_tracker.record_new_coverage(message.message_id, result.new_paths_covered)
    
    def _analyze_state_transitions(self, message: ParsedMessage, state_changes: List[str]):
        """Analyze state transitions for anomalies"""
        message_key = f"{message.message_type.value}_{message.round_number}"
        
        if message_key not in self.state_transitions:
            self.state_transitions[message_key] = []
        
        self.state_transitions[message_key].extend(state_changes)
        
        # Detect unusual state transition patterns
        if len(self.state_transitions[message_key]) > 10:
            # Keep only recent transitions
            self.state_transitions[message_key] = self.state_transitions[message_key][-10:]
    
    def _update_state_transitions(self, message: ParsedMessage, result: ExecutionResult):
        """Update state transition tracking"""
        if result.state_changes:
            transition_key = f"{message.sender_id}_{message.round_number}"
            if transition_key not in self.state_transitions:
                self.state_transitions[transition_key] = []
            self.state_transitions[transition_key].extend(result.state_changes)
    
    def _update_coverage_metrics(self, result: ExecutionResult):
        """Update overall coverage tracking"""
        if result.coverage_metrics:
            self.coverage_tracker.update_metrics(result.coverage_metrics)
        
        # Update fuzzing state coverage percentage
        self.fuzzing_state.coverage_percentage = self.coverage_tracker.get_total_coverage()
    
    def _update_fuzzing_state(self, result: ExecutionResult):
        """Update overall fuzzing state"""
        self.fuzzing_state.messages_executed += 1
        
        if result.fault_detected:
            self.fuzzing_state.faults_found += 1
        
        # Update coverage percentage
        if result.coverage_metrics:
            line_cov = result.coverage_metrics.get('line_coverage', 0)
            self.fuzzing_state.coverage_percentage = max(
                self.fuzzing_state.coverage_percentage, line_cov
            )
    
    def _create_empty_result(self) -> ExecutionResult:
        """Create empty execution result for failed iterations"""
        return ExecutionResult(
            message_id="",
            execution_time=0.0,
            success=False,
            state_changes=[],
            coverage_metrics={},
            fault_detected=False,
            error_message="No message selected for execution"
        )
    
    def get_fuzzing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive fuzzing statistics"""
        return {
            'current_state': {
                'iteration': self.fuzzing_state.iteration,
                'messages_executed': self.fuzzing_state.messages_executed,
                'faults_found': self.fuzzing_state.faults_found,
                'coverage_percentage': self.fuzzing_state.coverage_percentage,
                'runtime': self.fuzzing_state.runtime
            },
            'execution_stats': {
                'total_executions': len(self.execution_history),
                'successful_executions': sum(1 for r in self.execution_history if r.success),
                'fault_rate': self.fuzzing_state.faults_found / max(1, self.fuzzing_state.messages_executed),
                'average_execution_time': np.mean([r.execution_time for r in self.execution_history]) if self.execution_history else 0.0
            },
            'coverage_stats': self.coverage_tracker.get_statistics(),
            'fault_stats': self.fault_detector.get_statistics(),
            'state_transitions': len(self.state_transitions)
        }
    
    def reset_state(self):
        """Reset fuzzing state for new campaign"""
        self.fuzzing_state = FuzzingState(
            iteration=0,
            total_messages=0,
            clusters_count=0,
            messages_executed=0,
            faults_found=0,
            coverage_percentage=0.0
        )
        self.execution_history.clear()
        self.state_transitions.clear()
        self.coverage_tracker.reset()
        self.fault_detector.reset()


class CoverageTracker:
    """Tracks code coverage metrics across executions"""
    
    def __init__(self):
        self.total_coverage = {
            'line_coverage': 0.0,
            'branch_coverage': 0.0,
            'function_coverage': 0.0
        }
        self.coverage_history = []
        self.new_paths_total = 0
    
    def update_metrics(self, metrics: Dict[str, float]):
        """Update coverage metrics"""
        for key in self.total_coverage:
            if key in metrics:
                self.total_coverage[key] = max(self.total_coverage[key], metrics[key])
        
        self.coverage_history.append(metrics.copy())
    
    def record_new_coverage(self, message_id: str, new_paths: int):
        """Record new paths discovered"""
        self.new_paths_total += new_paths
    
    def get_total_coverage(self) -> float:
        """Get overall coverage percentage"""
        return float(np.mean(list(self.total_coverage.values())))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get coverage statistics"""
        return {
            'total_coverage': self.total_coverage,
            'new_paths_discovered': self.new_paths_total,
            'coverage_trend': self.coverage_history[-10:] if self.coverage_history else []
        }
    
    def reset(self):
        """Reset coverage tracking"""
        self.total_coverage = {'line_coverage': 0.0, 'branch_coverage': 0.0, 'function_coverage': 0.0}
        self.coverage_history.clear()
        self.new_paths_total = 0


class FaultDetector:
    """Detects and categorizes faults found during execution"""
    
    def __init__(self):
        self.faults = []
        self.fault_categories = defaultdict(int)
    
    def record_fault(self, message: ParsedMessage, result: ExecutionResult):
        """Record a fault discovered during execution"""
        fault_info = {
            'message_id': message.message_id,
            'message_type': message.message_type.value,
            'round_number': message.round_number,
            'timestamp': time.time(),
            'error_message': result.error_message,
            'success': result.success
        }
        
        self.faults.append(fault_info)
        
        # Categorize fault
        category = self._categorize_fault(result)
        self.fault_categories[category] += 1
    
    def _categorize_fault(self, result: ExecutionResult) -> str:
        """Categorize the type of fault"""
        if not result.success and result.error_message:
            error_msg = result.error_message.lower()
            if 'timeout' in error_msg:
                return 'timeout'
            elif 'crash' in error_msg:
                return 'crash'
            elif 'deadlock' in error_msg:
                return 'deadlock'
            elif 'consensus' in error_msg:
                return 'consensus_failure'
            else:
                return 'execution_error'
        elif result.fault_detected:
            return 'logic_fault'
        else:
            return 'unknown'
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get fault detection statistics"""
        return {
            'total_faults': len(self.faults),
            'fault_categories': dict(self.fault_categories),
            'recent_faults': self.faults[-5:] if self.faults else []
        }
    
    def reset(self):
        """Reset fault detection"""
        self.faults.clear()
        self.fault_categories.clear()