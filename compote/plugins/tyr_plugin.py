"""
Tyr Plugin for COMPOTE

Integrates COMPOTE with the Tyr fuzzing framework by:
- Providing high-priority message seeds to Tyr
- Receiving execution feedback from Tyr
- Sharing coverage and state data
"""

import time
import json
import subprocess
import threading
import tempfile
import os
from typing import Dict, List, Optional, Any
import logging

from .base_plugin import PluginInterface, FrameworkAdapter
from ..core.types import ParsedMessage, ExecutionResult, MessageType, NodeRole


class TyrAdapter(FrameworkAdapter):
    """Adapter for communicating with Tyr framework"""
    
    def __init__(self):
        super().__init__("Tyr")
        self.tyr_process = None
        self.input_pipe = None
        self.output_pipe = None
        self.working_dir = None
        self.tyr_executable = "tyr"  # Path to Tyr executable
        
    def connect(self, connection_params: Dict[str, Any]) -> bool:
        """Connect to Tyr framework by starting process"""
        try:
            self.tyr_executable = connection_params.get('tyr_path', 'tyr')
            self.working_dir = connection_params.get('working_dir', '/tmp')
            
            # Create named pipes for communication
            self.input_pipe = os.path.join(self.working_dir, 'compote_to_tyr.pipe')
            self.output_pipe = os.path.join(self.working_dir, 'tyr_to_compote.pipe')
            
            # Create pipes if they don't exist
            if not os.path.exists(self.input_pipe):
                os.mkfifo(self.input_pipe)
            if not os.path.exists(self.output_pipe):
                os.mkfifo(self.output_pipe)
            
            # Start Tyr process with COMPOTE integration
            tyr_args = [
                self.tyr_executable,
                '--compote-mode',
                '--input-pipe', self.input_pipe,
                '--output-pipe', self.output_pipe,
                '--working-dir', self.working_dir
            ]
            
            # Add any additional Tyr-specific arguments
            if 'tyr_args' in connection_params:
                tyr_args.extend(connection_params['tyr_args'])
            
            self.tyr_process = subprocess.Popen(
                tyr_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.working_dir
            )
            
            # Wait for Tyr to initialize
            time.sleep(2.0)
            
            if self.tyr_process.poll() is None:
                logging.info("Successfully started Tyr process")
                return True
            else:
                logging.error("Tyr process failed to start")
                return False
            
        except Exception as e:
            logging.error(f"Failed to connect to Tyr: {e}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from Tyr framework"""
        try:
            if self.tyr_process:
                # Send shutdown signal
                self._send_command({'type': 'shutdown'})
                
                # Wait for graceful shutdown
                try:
                    self.tyr_process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    self.tyr_process.kill()
                    self.tyr_process.wait()
                
                self.tyr_process = None
            
            # Clean up pipes
            if self.input_pipe and os.path.exists(self.input_pipe):
                os.unlink(self.input_pipe)
            if self.output_pipe and os.path.exists(self.output_pipe):
                os.unlink(self.output_pipe)
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to disconnect from Tyr: {e}")
            return False
    
    def send_message(self, message: ParsedMessage) -> bool:
        """Send message to Tyr"""
        try:
            command = {
                'type': 'execute_message',
                'message': {
                    'message_id': message.message_id,
                    'message_type': message.message_type.value,
                    'round_number': message.round_number,
                    'view_number': message.view_number,
                    'block_height': message.block_height,
                    'sender_id': message.sender_id,
                    'sender_role': message.sender_role.value,
                    'timestamp': message.timestamp,
                    'payload_hash': message.payload_hash,
                    'signature': message.signature,
                    'additional_fields': message.additional_fields
                }
            }
            
            return self._send_command(command)
            
        except Exception as e:
            logging.error(f"Failed to send message to Tyr: {e}")
            return False
    
    def receive_result(self) -> Optional[ExecutionResult]:
        """Receive execution result from Tyr"""
        try:
            response = self._receive_response()
            if not response or response.get('type') != 'execution_result':
                return None
            
            result_data = response.get('result', {})
            
            return ExecutionResult(
                message_id=result_data.get('message_id', ''),
                execution_time=result_data.get('execution_time', 0.0),
                success=result_data.get('success', False),
                state_changes=result_data.get('state_changes', []),
                coverage_metrics=result_data.get('coverage_metrics', {}),
                fault_detected=result_data.get('fault_detected', False),
                error_message=result_data.get('error_message'),
                new_paths_covered=result_data.get('new_paths_covered', 0)
            )
            
        except Exception as e:
            logging.error(f"Failed to receive result from Tyr: {e}")
            return None
    
    def _send_command(self, command: Dict[str, Any]) -> bool:
        """Send command to Tyr through input pipe"""
        try:
            if not self.input_pipe:
                return False
            
            command_json = json.dumps(command) + '\n'
            
            with open(self.input_pipe, 'w') as pipe:
                pipe.write(command_json)
                pipe.flush()
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to send command to Tyr: {e}")
            return False
    
    def _receive_response(self) -> Optional[Dict[str, Any]]:
        """Receive response from Tyr through output pipe"""
        try:
            if not self.output_pipe:
                return None
            
            with open(self.output_pipe, 'r') as pipe:
                line = pipe.readline().strip()
                if line:
                    return json.loads(line)
            
            return None
            
        except Exception as e:
            logging.error(f"Failed to receive response from Tyr: {e}")
            return None


class TyrPlugin(PluginInterface):
    """
    Tyr integration plugin for COMPOTE.
    
    Provides seamless integration with the Tyr fuzzing framework by:
    - Supplying high-priority message seeds
    - Collecting execution feedback
    - Sharing coverage information
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("Tyr", config)
        self.adapter = TyrAdapter()
        self.compote_engine = None
        self.result_monitor_thread = None
        self.stop_monitoring = threading.Event()
        self.logger = logging.getLogger(__name__)
        
        # Plugin-specific configuration
        self.batch_size = self.config.get('batch_size', 5)
        self.monitoring_interval = self.config.get('monitoring_interval', 0.5)
        self.result_timeout = self.config.get('result_timeout', 30.0)
        
        # Message tracking
        self.active_executions = {}
        self.completed_results = {}
    
    def initialize(self) -> bool:
        """Initialize Tyr plugin"""
        try:
            self.logger.info("Initializing Tyr plugin...")
            
            # Connect to Tyr framework
            connection_params = {
                'tyr_path': self.config.get('tyr_path', 'tyr'),
                'working_dir': self.config.get('working_dir', '/tmp/compote_tyr'),
                'tyr_args': self.config.get('tyr_args', [])
            }
            
            # Create working directory
            working_dir = connection_params['working_dir']
            os.makedirs(working_dir, exist_ok=True)
            
            if not self.adapter.connect(connection_params):
                self.logger.error("Failed to connect to Tyr framework")
                return False
            
            # Start result monitoring thread
            self.stop_monitoring.clear()
            self.result_monitor_thread = threading.Thread(target=self._monitor_results, daemon=True)
            self.result_monitor_thread.start()
            
            self.is_active = True
            self.logger.info("Tyr plugin initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Tyr plugin: {e}")
            return False
    
    def set_compote_engine(self, engine):
        """Set reference to COMPOTE engine for seed generation"""
        self.compote_engine = engine
    
    def provide_seed_messages(self, count: int = 10) -> List[ParsedMessage]:
        """Provide high-priority seed messages to Tyr"""
        if not self.compote_engine:
            self.logger.warning("No COMPOTE engine reference - cannot provide seeds")
            return []
        
        try:
            # Generate seed batch file for Tyr
            seed_messages = self._generate_seed_batch(count)
            
            if seed_messages:
                # Send batch to Tyr
                batch_command = {
                    'type': 'load_seed_batch',
                    'count': len(seed_messages),
                    'seeds': [self._message_to_dict(msg) for msg in seed_messages]
                }
                
                if self.adapter._send_command(batch_command):
                    self.update_statistics('messages_provided', len(seed_messages))
                    self.logger.info(f"Provided {len(seed_messages)} seed messages to Tyr")
                    return seed_messages
            
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to provide seed messages: {e}")
            return []
    
    def _generate_seed_batch(self, count: int) -> List[ParsedMessage]:
        """Generate a batch of high-priority seed messages"""
        seed_messages = []
        
        if not self.compote_engine.current_clusters:
            return seed_messages
        
        # Distribute seeds across clusters based on priority
        cluster_priorities = {}
        for cluster_id, cluster in self.compote_engine.current_clusters.items():
            if cluster_id >= 0 and cluster.priority_scores:
                avg_priority = sum(cluster.priority_scores.values()) / len(cluster.priority_scores)
                cluster_priorities[cluster_id] = avg_priority
        
        # Sort clusters by priority
        sorted_clusters = sorted(cluster_priorities.items(), key=lambda x: x[1], reverse=True)
        
        # Select messages from top clusters
        seeds_per_cluster = max(1, count // len(sorted_clusters)) if sorted_clusters else 0
        
        for cluster_id, _ in sorted_clusters:
            cluster = self.compote_engine.current_clusters[cluster_id]
            
            # Get top messages from this cluster
            top_messages = self.compote_engine.priority_calculator.get_top_priority_messages(
                cluster_id, top_k=seeds_per_cluster
            )
            
            for message_id, priority in top_messages:
                if message_id in self.compote_engine.parsed_messages:
                    message = self.compote_engine.parsed_messages[message_id]
                    seed_messages.append(message)
                    
                    if len(seed_messages) >= count:
                        break
            
            if len(seed_messages) >= count:
                break
        
        return seed_messages[:count]
    
    def _message_to_dict(self, message: ParsedMessage) -> Dict[str, Any]:
        """Convert ParsedMessage to dictionary for Tyr"""
        return {
            'message_id': message.message_id,
            'message_type': message.message_type.value,
            'round_number': message.round_number,
            'view_number': message.view_number,
            'block_height': message.block_height,
            'sender_id': message.sender_id,
            'sender_role': message.sender_role.value,
            'timestamp': message.timestamp,
            'payload_hash': message.payload_hash,
            'signature': message.signature,
            'additional_fields': message.additional_fields
        }
    
    def execute_message(self, message: ParsedMessage) -> ExecutionResult:
        """Execute message through Tyr framework"""
        try:
            # Send execution request
            if not self.adapter.send_message(message):
                return self._create_error_result(message.message_id, "Failed to send to Tyr")
            
            # Track execution
            self.active_executions[message.message_id] = {
                'start_time': time.time(),
                'message': message
            }
            
            # Wait for result
            start_time = time.time()
            while time.time() - start_time < self.result_timeout:
                if message.message_id in self.completed_results:
                    result = self.completed_results.pop(message.message_id)
                    self.active_executions.pop(message.message_id, None)
                    self.update_statistics('messages_executed')
                    if result.fault_detected:
                        self.update_statistics('faults_found')
                    return result
                
                time.sleep(0.1)
            
            # Timeout
            self.active_executions.pop(message.message_id, None)
            return self._create_error_result(message.message_id, "Execution timeout")
            
        except Exception as e:
            self.logger.error(f"Failed to execute message through Tyr: {e}")
            return self._create_error_result(message.message_id, str(e))
    
    def receive_feedback(self, result: ExecutionResult) -> bool:
        """Receive execution feedback from Tyr"""
        try:
            if not self.compote_engine:
                return False
            
            # Update COMPOTE's priority calculator
            self.compote_engine.priority_calculator.update_execution_result(
                result.message_id, result
            )
            
            self.update_statistics('feedback_received')
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process feedback: {e}")
            return False
    
    def _monitor_results(self):
        """Background thread for monitoring execution results"""
        while not self.stop_monitoring.is_set():
            try:
                result = self.adapter.receive_result()
                if result:
                    # Store result for retrieval
                    self.completed_results[result.message_id] = result
                    
                    # Process feedback
                    self.receive_feedback(result)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.debug(f"Result monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _create_error_result(self, message_id: str, error_message: str) -> ExecutionResult:
        """Create error execution result"""
        return ExecutionResult(
            message_id=message_id,
            execution_time=0.0,
            success=False,
            state_changes=[],
            coverage_metrics={},
            fault_detected=False,
            error_message=error_message
        )
    
    def get_tyr_status(self) -> Dict[str, Any]:
        """Get Tyr framework status"""
        try:
            status_command = {'type': 'get_status'}
            if self.adapter._send_command(status_command):
                response = self.adapter._receive_response()
                if response and response.get('type') == 'status_response':
                    return response.get('status', {})
            
            return {'status': 'unknown'}
            
        except Exception as e:
            self.logger.error(f"Failed to get Tyr status: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def shutdown(self) -> bool:
        """Shutdown Tyr plugin"""
        try:
            self.logger.info("Shutting down Tyr plugin...")
            
            # Stop result monitoring
            self.stop_monitoring.set()
            if self.result_monitor_thread and self.result_monitor_thread.is_alive():
                self.result_monitor_thread.join(timeout=5.0)
            
            # Disconnect from Tyr
            self.adapter.disconnect()
            
            self.is_active = False
            self.logger.info("Tyr plugin shutdown complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to shutdown Tyr plugin: {e}")
            return False
    
    def get_detailed_statistics(self) -> Dict[str, Any]:
        """Get detailed plugin statistics"""
        base_stats = self.get_statistics()
        
        base_stats.update({
            'active_executions': len(self.active_executions),
            'completed_results': len(self.completed_results),
            'tyr_status': self.get_tyr_status(),
            'batch_size': self.batch_size,
            'monitoring_interval': self.monitoring_interval
        })
        
        return base_stats