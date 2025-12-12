"""
Main COMPOTE Fuzzing Engine

Orchestrates all algorithms into a unified fuzzing framework:
- Integrates parsing, feature extraction, clustering, priority calculation, and state analysis
- Provides both standalone and plugin operation modes
- Implements the complete feedback loop for continuous refinement
"""

import time
import json
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading

from ..core.types import (
    RawMessage, ParsedMessage, MessageFeatures, MessageCluster, 
    PriorityScore, ExecutionResult, FuzzingState
)
from ..core.parser import MessageParser
from ..core.feature_extractor import FeatureExtractor
from ..core.clustering import ContextClustering
from ..core.priority_calculator import PriorityCalculator
from ..core.state_analyzer import StateAnalyzer, MutationStrategy, ExecutionEnvironment


class CompoteFuzzer:
    """
    Main COMPOTE fuzzing engine that orchestrates all algorithms.
    
    Provides a complete consensus protocol fuzzing solution with:
    - Automated message parsing and feature extraction
    - Context-aware clustering and prioritization
    - Intelligent mutation and execution
    - Continuous feedback-driven refinement
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize COMPOTE fuzzing engine.
        
        Args:
            config: Configuration dictionary for all components
        """
        self.config = config or self._get_default_config()
        
        # Initialize all algorithm components
        self.parser = MessageParser()
        self.feature_extractor = FeatureExtractor(
            normalize=self.config.get('normalize_features', True)
        )
        self.clustering = ContextClustering(
            eps=self.config.get('clustering_eps', 0.5),
            min_samples=self.config.get('clustering_min_samples', 3),
            scale_features=self.config.get('scale_features', True)
        )
        self.priority_calculator = PriorityCalculator(
            alpha=self.config.get('priority_alpha', 0.3),
            beta=self.config.get('priority_beta', 0.4), 
            gamma=self.config.get('priority_gamma', 0.3),
            fault_threshold=self.config.get('fault_threshold', 0.1),
            similarity_threshold=self.config.get('similarity_threshold', 0.2),
            time_threshold=self.config.get('time_threshold', 3600.0)
        )
        
        # Initialize state analyzer with strategies
        mutation_strategy = MutationStrategy(
            timestamp_variance=self.config.get('timestamp_variance', 0.1),
            round_variance=self.config.get('round_variance', 2),
            view_variance=self.config.get('view_variance', 1),
            enabled_mutations=self.config.get('enabled_mutations', 
                ['timestamp', 'round', 'view', 'payload', 'sender'])
        )
        
        execution_env = ExecutionEnvironment(
            execution_timeout=self.config.get('execution_timeout', 30.0),
            max_retries=self.config.get('max_retries', 3),
            simulation_mode=self.config.get('simulation_mode', False)
        )
        
        self.state_analyzer = StateAnalyzer(mutation_strategy, execution_env)
        
        # Fuzzing state
        self.is_running = False
        self.current_clusters: Dict[int, MessageCluster] = {}
        self.message_features: Dict[str, MessageFeatures] = {}
        self.parsed_messages: Dict[str, ParsedMessage] = {}
        self.raw_messages: List[RawMessage] = []
        
        # Threading for concurrent operations
        self.executor = ThreadPoolExecutor(max_workers=self.config.get('max_workers', 4))
        self.stop_event = threading.Event()
        
        # Callbacks and hooks
        self.progress_callback: Optional[Callable] = None
        self.result_callback: Optional[Callable] = None
        self.fault_callback: Optional[Callable] = None
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Statistics and metrics
        self.start_time = 0.0
        self.total_iterations = 0
        self.session_stats = {
            'messages_processed': 0,
            'clusters_created': 0,
            'faults_discovered': 0,
            'execution_errors': 0
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            # Feature extraction
            'normalize_features': True,
            
            # Clustering
            'clustering_eps': 0.5,
            'clustering_min_samples': 3,
            'scale_features': True,
            
            # Priority calculation
            'priority_alpha': 0.3,  # similarity weight
            'priority_beta': 0.4,   # fault weight
            'priority_gamma': 0.3,  # coverage weight
            'fault_threshold': 0.1,
            'similarity_threshold': 0.2,
            'time_threshold': 3600.0,
            
            # Mutation
            'timestamp_variance': 0.1,
            'round_variance': 2,
            'view_variance': 1,
            'enabled_mutations': ['timestamp', 'round', 'view', 'payload', 'sender'],
            
            # Execution
            'execution_timeout': 30.0,
            'max_retries': 3,
            'simulation_mode': False,
            
            # Engine
            'max_workers': 4,
            'max_iterations': 1000,
            'save_interval': 100,
            'auto_optimize': True
        }
    
    def load_messages(self, messages: List[Union[RawMessage, Dict[str, Any]]], 
                     format_type: str = 'json') -> int:
        """
        Load raw messages into the fuzzer.
        
        Args:
            messages: List of raw messages or message dictionaries
            format_type: Format of messages ('json', 'binary', 'protobuf', 'custom')
            
        Returns:
            Number of successfully loaded messages
        """
        loaded_count = 0
        
        for msg_data in messages:
            try:
                if isinstance(msg_data, RawMessage):
                    raw_msg = msg_data
                else:
                    # Convert dictionary to RawMessage
                    if isinstance(msg_data, dict):
                        raw_data = json.dumps(msg_data).encode('utf-8')
                    else:
                        raw_data = str(msg_data).encode('utf-8')
                    
                    raw_msg = RawMessage(
                        data=raw_data,
                        timestamp=time.time(),
                        source_id=msg_data.get('source_id', 'unknown') if isinstance(msg_data, dict) else 'unknown'
                    )
                
                self.raw_messages.append(raw_msg)
                loaded_count += 1
                
            except Exception as e:
                self.logger.warning(f"Failed to load message: {e}")
                continue
        
        self.session_stats['messages_processed'] = loaded_count
        self.logger.info(f"Loaded {loaded_count} messages for fuzzing")
        return loaded_count
    
    def initialize_seed_pool(self) -> bool:
        """
        Initialize the seed pool by processing loaded messages.
        
        Returns:
            True if initialization successful
        """
        if not self.raw_messages:
            self.logger.error("No messages loaded. Call load_messages() first.")
            return False
        
        try:
            # Step 1: Parse all raw messages
            self.logger.info("Parsing raw messages...")
            parsed_messages = []
            
            for raw_msg in self.raw_messages:
                try:
                    parsed_msg = self.parser.parse(raw_msg)
                    parsed_messages.append(parsed_msg)
                    self.parsed_messages[parsed_msg.message_id] = parsed_msg
                except Exception as e:
                    self.logger.warning(f"Failed to parse message {raw_msg.message_id}: {e}")
                    continue
            
            if not parsed_messages:
                self.logger.error("No messages successfully parsed")
                return False
            
            # Step 2: Extract features
            self.logger.info("Extracting features...")
            message_features = self.feature_extractor.extract_features(parsed_messages)
            
            for features in message_features:
                self.message_features[features.message_id] = features
            
            # Step 3: Perform clustering
            self.logger.info("Performing context clustering...")
            self.current_clusters = self.clustering.fit_predict(message_features)
            self.session_stats['clusters_created'] = len(self.current_clusters)
            
            # Step 4: Calculate initial priorities
            self.logger.info("Calculating initial priorities...")
            priorities = self.priority_calculator.calculate_priorities(
                self.current_clusters, self.message_features
            )
            
            self.logger.info(f"Initialized seed pool with {len(parsed_messages)} messages, "
                           f"{len(self.current_clusters)} clusters")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize seed pool: {e}")
            return False
    
    def start_fuzzing(self, max_iterations: Optional[int] = None) -> bool:
        """
        Start the main fuzzing loop.
        
        Args:
            max_iterations: Maximum number of iterations (None for unlimited)
            
        Returns:
            True if fuzzing started successfully
        """
        if not self.current_clusters:
            self.logger.error("Seed pool not initialized. Call initialize_seed_pool() first.")
            return False
        
        if self.is_running:
            self.logger.warning("Fuzzing is already running")
            return False
        
        max_iterations = max_iterations or self.config.get('max_iterations', 1000)
        # Ensure max_iterations is an int (it's guaranteed to be non-None after the 'or' operation)
        max_iterations = int(max_iterations) if max_iterations is not None else 1000
        
        self.logger.info(f"Starting COMPOTE fuzzing with max {max_iterations} iterations")
        self.is_running = True
        self.start_time = time.time()
        self.stop_event.clear()
        
        try:
            self._fuzzing_loop(max_iterations)
            return True
        except Exception as e:
            self.logger.error(f"Fuzzing failed: {e}")
            self.is_running = False
            return False
    
    def _fuzzing_loop(self, max_iterations: int):
        """Main fuzzing loop implementing the COMPOTE algorithm"""
        
        for iteration in range(max_iterations):
            if self.stop_event.is_set():
                self.logger.info("Fuzzing stopped by user request")
                break
            
            try:
                # Run single fuzzing iteration
                result = self.state_analyzer.run_fuzzing_iteration(
                    self.current_clusters, self.priority_calculator
                )
                
                # Process iteration result
                self._process_iteration_result(result, iteration)
                
                # Update statistics
                self.total_iterations += 1
                if result.fault_detected:
                    self.session_stats['faults_discovered'] += 1
                if not result.success:
                    self.session_stats['execution_errors'] += 1
                
                # Periodic tasks
                if iteration % self.config.get('save_interval', 100) == 0:
                    self._periodic_maintenance(iteration)
                
                # Auto-optimization
                if self.config.get('auto_optimize', True) and iteration % 200 == 0:
                    self._auto_optimize()
                
                # Progress callback
                if self.progress_callback:
                    self.progress_callback(iteration, max_iterations, result)
                
            except Exception as e:
                self.logger.error(f"Error in fuzzing iteration {iteration}: {e}")
                continue
        
        self.is_running = False
        self.logger.info(f"Fuzzing completed after {self.total_iterations} iterations")
    
    def _process_iteration_result(self, result: ExecutionResult, iteration: int):
        """Process the result of a fuzzing iteration"""
        
        # Log significant results
        if result.fault_detected:
            self.logger.warning(f"Iteration {iteration}: Fault detected in message {result.message_id}")
            if self.fault_callback:
                self.fault_callback(result)
        
        if result.new_paths_covered > 0:
            self.logger.info(f"Iteration {iteration}: Discovered {result.new_paths_covered} new paths")
        
        # Result callback
        if self.result_callback:
            self.result_callback(result)
    
    def _periodic_maintenance(self, iteration: int):
        """Perform periodic maintenance tasks"""
        self.logger.info(f"Iteration {iteration}: Performing maintenance...")
        
        # Re-cluster if needed (every 500 iterations)
        if iteration % 500 == 0 and iteration > 0:
            self._recluster_messages()
        
        # Update priority weights based on performance
        if iteration % 300 == 0 and iteration > 0:
            self._update_priority_weights()
        
        # Save current state
        self.save_state(f"compote_state_iter_{iteration}.json")
    
    def _recluster_messages(self):
        """Re-cluster messages with updated features"""
        try:
            self.logger.info("Re-clustering messages...")
            
            # Get current message features
            feature_list = list(self.message_features.values())
            
            if feature_list:
                # Perform new clustering
                new_clusters = self.clustering.fit_predict(feature_list)
                
                # Update clusters
                self.current_clusters = new_clusters
                
                # Recalculate priorities
                self.priority_calculator.calculate_priorities(
                    self.current_clusters, self.message_features, force_recalculate=True
                )
                
                self.logger.info(f"Re-clustering complete: {len(new_clusters)} clusters")
        
        except Exception as e:
            self.logger.error(f"Re-clustering failed: {e}")
    
    def _update_priority_weights(self):
        """Update priority calculation weights based on performance"""
        try:
            # Get current performance metrics
            stats = self.state_analyzer.get_fuzzing_statistics()
            fault_rate = stats['execution_stats']['fault_rate']
            
            # Adjust weights based on fault discovery rate
            if fault_rate > 0.1:  # High fault rate - increase fault weight
                new_beta = min(0.6, self.priority_calculator.beta + 0.1)
                new_alpha = max(0.2, self.priority_calculator.alpha - 0.05)
                new_gamma = 1.0 - new_alpha - new_beta
            elif fault_rate < 0.02:  # Low fault rate - increase similarity weight
                new_alpha = min(0.5, self.priority_calculator.alpha + 0.1)
                new_beta = max(0.2, self.priority_calculator.beta - 0.05)
                new_gamma = 1.0 - new_alpha - new_beta
            else:
                return  # No adjustment needed
            
            self.priority_calculator.update_weights(new_alpha, new_beta, new_gamma)
            self.logger.info(f"Updated priority weights: α={new_alpha:.2f}, β={new_beta:.2f}, γ={new_gamma:.2f}")
        
        except Exception as e:
            self.logger.error(f"Weight update failed: {e}")
    
    def _auto_optimize(self):
        """Automatically optimize clustering and other parameters"""
        try:
            self.logger.info("Running auto-optimization...")
            
            # Optimize clustering parameters
            feature_list = list(self.message_features.values())
            if feature_list:
                optimal_params = self.clustering.optimize_parameters(feature_list)
                self.logger.info(f"Optimal clustering params: {optimal_params}")
        
        except Exception as e:
            self.logger.error(f"Auto-optimization failed: {e}")
    
    def stop_fuzzing(self):
        """Stop the fuzzing process"""
        self.logger.info("Stopping fuzzing...")
        self.stop_event.set()
        self.is_running = False
    
    def save_state(self, file_path: str) -> bool:
        """
        Save current fuzzer state to file.
        
        Args:
            file_path: Path to save state file
            
        Returns:
            True if save successful
        """
        try:
            state_data = {
                'config': self.config,
                'session_stats': self.session_stats,
                'total_iterations': self.total_iterations,
                'runtime': time.time() - self.start_time if self.start_time > 0 else 0,
                'clustering_stats': self.clustering.get_clustering_stats(),
                'priority_stats': self.priority_calculator.get_calculation_stats(),
                'fuzzing_stats': self.state_analyzer.get_fuzzing_statistics(),
                'cluster_count': len(self.current_clusters),
                'message_count': len(self.message_features)
            }
            
            with open(file_path, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
            
            self.logger.info(f"State saved to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
            return False
    
    def load_state(self, file_path: str) -> bool:
        """
        Load fuzzer state from file.
        
        Args:
            file_path: Path to state file
            
        Returns:
            True if load successful
        """
        try:
            with open(file_path, 'r') as f:
                state_data = json.load(f)
            
            # Restore configuration and statistics
            self.config.update(state_data.get('config', {}))
            self.session_stats = state_data.get('session_stats', {})
            self.total_iterations = state_data.get('total_iterations', 0)
            
            self.logger.info(f"State loaded from {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
            return False
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive fuzzing report"""
        runtime = time.time() - self.start_time if self.start_time > 0 else 0
        
        return {
            'summary': {
                'total_runtime': runtime,
                'total_iterations': self.total_iterations,
                'messages_processed': self.session_stats['messages_processed'],
                'clusters_created': self.session_stats['clusters_created'],
                'faults_discovered': self.session_stats['faults_discovered'],
                'execution_errors': self.session_stats['execution_errors'],
                'is_running': self.is_running
            },
            'algorithm_stats': {
                'parsing': self.parser.get_parsing_stats(),
                'feature_extraction': self.feature_extractor.get_stats(),
                'clustering': self.clustering.get_clustering_stats(),
                'priority_calculation': self.priority_calculator.get_calculation_stats(),
                'state_analysis': self.state_analyzer.get_fuzzing_statistics()
            },
            'performance': {
                'iterations_per_second': self.total_iterations / max(1, runtime),
                'fault_discovery_rate': self.session_stats['faults_discovered'] / max(1, self.total_iterations),
                'success_rate': 1.0 - (self.session_stats['execution_errors'] / max(1, self.total_iterations))
            },
            'current_state': {
                'active_clusters': len(self.current_clusters),
                'cached_priorities': len(self.priority_calculator.priority_cache),
                'historical_records': len(self.priority_calculator.historical_records)
            }
        }
    
    def set_progress_callback(self, callback: Callable):
        """Set callback for progress updates"""
        self.progress_callback = callback
    
    def set_result_callback(self, callback: Callable):
        """Set callback for iteration results"""
        self.result_callback = callback
    
    def set_fault_callback(self, callback: Callable):
        """Set callback for fault detection"""
        self.fault_callback = callback
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = self.config.get('log_level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.is_running:
            self.stop_fuzzing()
        self.executor.shutdown(wait=True)