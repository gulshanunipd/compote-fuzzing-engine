"""
LOKI Plugin for COMPOTE

Integrates COMPOTE with the LOKI fuzzing framework by:
- Providing high-priority message seeds to LOKI
- Receiving execution feedback from LOKI
- Sharing coverage and state data
"""

import time
import json
import socket
import threading
from typing import Dict, List, Optional, Any
import logging

from .base_plugin import PluginInterface, FrameworkAdapter
from ..core.types import ParsedMessage, ExecutionResult, MessageType, NodeRole


class LokiAdapter(FrameworkAdapter):
    """Adapter for communicating with LOKI framework"""
    
    def __init__(self):
        super().__init__("LOKI")
        self.socket = None
        self.host = "localhost"
        self.port = 9001
        self.timeout = 30.0
        
    def connect(self, connection_params: Dict[str, Any]) -> bool:
        """Connect to LOKI framework"""
        try:
            self.host = connection_params.get('host', self.host)
            self.port = connection_params.get('port', self.port)
            self.timeout = connection_params.get('timeout', self.timeout)
            
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            
            # Send handshake
            handshake = {
                'type': 'handshake',
                'client': 'COMPOTE',
                'version': '1.0',
                'capabilities': ['message_seeding', 'feedback_collection']
            }
            
            self._send_json(handshake)
            response = self._receive_json()
            
            return response and response.get('status') == 'connected'
            
        except Exception as e:
            logging.error(f"Failed to connect to LOKI: {e}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from LOKI framework"""
        try:
            if self.socket:
                disconnect_msg = {'type': 'disconnect', 'client': 'COMPOTE'}
                self._send_json(disconnect_msg)
                self.socket.close()
                self.socket = None
            return True
        except Exception as e:
            logging.error(f"Failed to disconnect from LOKI: {e}")
            return False
    
    def send_message(self, message: ParsedMessage) -> bool:
        """Send message to LOKI"""
        try:
            message_data = {
                'type': 'seed_message',
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
            
            self._send_json(message_data)
            return True
            
        except Exception as e:
            logging.error(f"Failed to send message to LOKI: {e}")
            return False
    
    def receive_result(self) -> Optional[ExecutionResult]:
        """Receive execution result from LOKI"""
        try:
            result_data = self._receive_json()
            if not result_data or result_data.get('type') != 'execution_result':
                return None
            
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
            logging.error(f"Failed to receive result from LOKI: {e}")
            return None
    
    def _send_json(self, data: Dict[str, Any]):
        """Send JSON data over socket"""
        if not self.socket:
            raise ConnectionError("Not connected to LOKI")
        
        json_data = json.dumps(data)
        message = f"{len(json_data)}\n{json_data}"
        self.socket.sendall(message.encode('utf-8'))
    
    def _receive_json(self) -> Optional[Dict[str, Any]]:
        """Receive JSON data from socket"""
        if not self.socket:
            raise ConnectionError("Not connected to LOKI")
        
        # Read message length
        length_str = ""
        while True:
            char = self.socket.recv(1).decode('utf-8')
            if char == '\n':
                break
            length_str += char
        
        if not length_str:
            return None
        
        # Read message data
        message_length = int(length_str)
        message_data = self.socket.recv(message_length).decode('utf-8')
        
        return json.loads(message_data)


class LokiPlugin(PluginInterface):
    """
    LOKI integration plugin for COMPOTE.
    
    Provides seamless integration with the LOKI fuzzing framework by:
    - Supplying high-priority message seeds
    - Collecting execution feedback
    - Sharing coverage information
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("LOKI", config)
        self.adapter = LokiAdapter()
        self.compote_engine = None
        self.feedback_thread = None
        self.stop_feedback = threading.Event()
        self.logger = logging.getLogger(__name__)
        
        # Plugin-specific configuration
        self.seed_batch_size = self.config.get('seed_batch_size', 5)
        self.feedback_interval = self.config.get('feedback_interval', 1.0)
        self.auto_reseed = self.config.get('auto_reseed', True)
        self.reseed_threshold = self.config.get('reseed_threshold', 10)
        
        # Message tracking
        self.sent_messages = {}
        self.pending_results = set()
    
    def initialize(self) -> bool:
        """Initialize LOKI plugin"""
        try:
            self.logger.info("Initializing LOKI plugin...")
            
            # Connect to LOKI framework
            connection_params = {
                'host': self.config.get('loki_host', 'localhost'),
                'port': self.config.get('loki_port', 9001),
                'timeout': self.config.get('connection_timeout', 30.0)
            }
            
            if not self.adapter.connect(connection_params):
                self.logger.error("Failed to connect to LOKI framework")
                return False
            
            # Start feedback collection thread
            self.stop_feedback.clear()
            self.feedback_thread = threading.Thread(target=self._feedback_loop, daemon=True)
            self.feedback_thread.start()
            
            self.is_active = True
            self.logger.info("LOKI plugin initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LOKI plugin: {e}")
            return False
    
    def set_compote_engine(self, engine):
        """Set reference to COMPOTE engine for seed generation"""
        self.compote_engine = engine
    
    def provide_seed_messages(self, count: int = 10) -> List[ParsedMessage]:
        """Provide high-priority seed messages to LOKI"""
        if not self.compote_engine:
            self.logger.warning("No COMPOTE engine reference - cannot provide seeds")
            return []
        
        try:
            # Get high-priority messages from COMPOTE clusters
            seed_messages = []
            
            for cluster_id, cluster in self.compote_engine.current_clusters.items():
                if cluster_id < 0:  # Skip noise clusters
                    continue
                
                # Get top priority messages from this cluster
                top_messages = self.compote_engine.priority_calculator.get_top_priority_messages(
                    cluster_id, top_k=min(3, count // len(self.compote_engine.current_clusters))
                )
                
                for message_id, priority in top_messages:
                    if message_id in self.compote_engine.parsed_messages:
                        message = self.compote_engine.parsed_messages[message_id]
                        seed_messages.append(message)
                        
                        # Track sent message
                        self.sent_messages[message_id] = {
                            'timestamp': time.time(),
                            'priority': priority,
                            'cluster_id': cluster_id
                        }
                
                if len(seed_messages) >= count:
                    break
            
            # Send messages to LOKI
            sent_count = 0
            for message in seed_messages[:count]:
                if self.adapter.send_message(message):
                    sent_count += 1
                    self.pending_results.add(message.message_id)
            
            self.update_statistics('messages_provided', sent_count)
            self.logger.info(f"Provided {sent_count} seed messages to LOKI")
            
            return seed_messages[:sent_count]
            
        except Exception as e:
            self.logger.error(f"Failed to provide seed messages: {e}")
            return []
    
    def execute_message(self, message: ParsedMessage) -> ExecutionResult:
        """Execute message through LOKI framework"""
        try:
            # Send message to LOKI
            if not self.adapter.send_message(message):
                return ExecutionResult(
                    message_id=message.message_id,
                    execution_time=0.0,
                    success=False,
                    state_changes=[],
                    coverage_metrics={},
                    fault_detected=False,
                    error_message="Failed to send message to LOKI"
                )
            
            # Wait for result (with timeout)
            start_time = time.time()
            timeout = self.config.get('execution_timeout', 30.0)
            
            while time.time() - start_time < timeout:
                result = self.adapter.receive_result()
                if result and result.message_id == message.message_id:
                    self.update_statistics('messages_executed')
                    if result.fault_detected:
                        self.update_statistics('faults_found')
                    return result
                
                time.sleep(0.1)
            
            # Timeout
            return ExecutionResult(
                message_id=message.message_id,
                execution_time=timeout,
                success=False,
                state_changes=[],
                coverage_metrics={},
                fault_detected=False,
                error_message="Execution timeout"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to execute message through LOKI: {e}")
            return ExecutionResult(
                message_id=message.message_id,
                execution_time=0.0,
                success=False,
                state_changes=[],
                coverage_metrics={},
                fault_detected=False,
                error_message=str(e)
            )
    
    def receive_feedback(self, result: ExecutionResult) -> bool:
        """Receive execution feedback from LOKI"""
        try:
            if not self.compote_engine:
                return False
            
            # Update COMPOTE's priority calculator with feedback
            self.compote_engine.priority_calculator.update_execution_result(
                result.message_id, result
            )
            
            # Remove from pending results
            self.pending_results.discard(result.message_id)
            
            self.update_statistics('feedback_received')
            
            # Auto-reseed if needed
            if (self.auto_reseed and 
                len(self.pending_results) < self.reseed_threshold):
                self._auto_reseed()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process feedback: {e}")
            return False
    
    def _feedback_loop(self):
        """Background thread for collecting feedback from LOKI"""
        while not self.stop_feedback.is_set():
            try:
                result = self.adapter.receive_result()
                if result:
                    self.receive_feedback(result)
                
                time.sleep(self.feedback_interval)
                
            except Exception as e:
                self.logger.debug(f"Feedback loop error: {e}")
                time.sleep(self.feedback_interval)
    
    def _auto_reseed(self):
        """Automatically provide new seeds when pending results are low"""
        try:
            new_seeds = self.provide_seed_messages(self.seed_batch_size)
            if new_seeds:
                self.logger.info(f"Auto-reseeded with {len(new_seeds)} messages")
        except Exception as e:
            self.logger.error(f"Auto-reseed failed: {e}")
    
    def shutdown(self) -> bool:
        """Shutdown LOKI plugin"""
        try:
            self.logger.info("Shutting down LOKI plugin...")
            
            # Stop feedback thread
            self.stop_feedback.set()
            if self.feedback_thread and self.feedback_thread.is_alive():
                self.feedback_thread.join(timeout=5.0)
            
            # Disconnect from LOKI
            self.adapter.disconnect()
            
            self.is_active = False
            self.logger.info("LOKI plugin shutdown complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to shutdown LOKI plugin: {e}")
            return False
    
    def get_detailed_statistics(self) -> Dict[str, Any]:
        """Get detailed plugin statistics"""
        base_stats = self.get_statistics()
        
        base_stats.update({
            'sent_messages': len(self.sent_messages),
            'pending_results': len(self.pending_results),
            'connection_status': 'connected' if self.adapter.socket else 'disconnected',
            'auto_reseed_enabled': self.auto_reseed,
            'seed_batch_size': self.seed_batch_size,
            'recent_seeds': list(self.sent_messages.keys())[-10:]  # Last 10 seeds
        })
        
        return base_stats