"""
Hyperledger Fabric Integration Module for COMPOTE

Provides comprehensive integration with Hyperledger Fabric v2.5 including:
- Docker container management
- gRPC communication with orderers and peers
- REST API integration for chaincode operations
- Consensus message extraction and injection
- Coverage tracking and fault detection
"""

from .fabric_integration import (
    FabricNetworkConfig,
    FabricDockerManager, 
    FabricGRPCClient,
    FabricRESTClient
)
from .fabric_plugin import FabricPlugin

__all__ = [
    'FabricNetworkConfig',
    'FabricDockerManager',
    'FabricGRPCClient', 
    'FabricRESTClient',
    'FabricPlugin'
]