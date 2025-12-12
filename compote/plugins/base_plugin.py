"""
Base Plugin Interface

Defines the common interface for integrating COMPOTE with other fuzzing frameworks
like LOKI and Tyr. Provides standardized methods for message generation, execution,
and feedback collection.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from ..core.types import ParsedMessage, ExecutionResult, MessageFeatures


class PluginInterface(ABC):
    """
    Abstract base class for COMPOTE plugins.
    
    Plugins allow COMPOTE to integrate with existing fuzzing frameworks
    by providing high-priority mutated messages and receiving feedback.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize plugin.
        
        Args:
            name: Plugin name
            config: Plugin configuration
        """
        self.name = name
        self.config = config or {}
        self.is_active = False
        self.statistics = {
            'messages_provided': 0,
            'messages_executed': 0,
            'faults_found': 0,
            'feedback_received': 0
        }
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the plugin and connect to target framework.
        
        Returns:
            True if initialization successful
        """
        pass
    
    @abstractmethod
    def provide_seed_messages(self, count: int = 10) -> List[ParsedMessage]:
        """
        Provide high-priority seed messages to the target framework.
        
        Args:
            count: Number of messages to provide
            
        Returns:
            List of prioritized ParsedMessages
        """
        pass
    
    @abstractmethod
    def execute_message(self, message: ParsedMessage) -> ExecutionResult:
        """
        Execute a message through the target framework.
        
        Args:
            message: Message to execute
            
        Returns:
            Execution result with coverage and fault data
        """
        pass
    
    @abstractmethod
    def receive_feedback(self, result: ExecutionResult) -> bool:
        """
        Receive execution feedback from the target framework.
        
        Args:
            result: Execution result with feedback data
            
        Returns:
            True if feedback processed successfully
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> bool:
        """
        Shutdown the plugin and cleanup resources.
        
        Returns:
            True if shutdown successful
        """
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get plugin statistics"""
        return {
            'name': self.name,
            'is_active': self.is_active,
            'statistics': self.statistics.copy(),
            'config': self.config
        }
    
    def update_statistics(self, stat_name: str, increment: int = 1):
        """Update plugin statistics"""
        if stat_name in self.statistics:
            self.statistics[stat_name] += increment


class FrameworkAdapter:
    """
    Base adapter for integrating with fuzzing frameworks.
    
    Provides common functionality for framework integration.
    """
    
    def __init__(self, framework_name: str):
        """
        Initialize framework adapter.
        
        Args:
            framework_name: Name of the target framework
        """
        self.framework_name = framework_name
        self.connection = None
        self.message_queue = []
        self.result_queue = []
    
    def connect(self, connection_params: Dict[str, Any]) -> bool:
        """
        Connect to the target framework.
        
        Args:
            connection_params: Connection parameters
            
        Returns:
            True if connection successful
        """
        # Override in subclasses
        return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from the target framework.
        
        Returns:
            True if disconnection successful
        """
        # Override in subclasses
        return False
    
    def send_message(self, message: ParsedMessage) -> bool:
        """
        Send message to framework.
        
        Args:
            message: Message to send
            
        Returns:
            True if send successful
        """
        # Override in subclasses
        return False
    
    def receive_result(self) -> Optional[ExecutionResult]:
        """
        Receive execution result from framework.
        
        Returns:
            ExecutionResult or None if no result available
        """
        # Override in subclasses
        return None