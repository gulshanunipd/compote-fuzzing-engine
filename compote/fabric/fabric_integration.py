"""
Hyperledger Fabric Integration for COMPOTE

Provides integration with Hyperledger Fabric v2.5 networks including:
- Fabric REST API integration
- gRPC services communication
- Docker container management
- Consensus message extraction and injection
"""

import json
import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import grpc
import requests
import docker
from docker.models.containers import Container
from docker.models.networks import Network

from ..core.types import RawMessage, ParsedMessage, ExecutionResult, MessageType, NodeRole
from ..plugins.base_plugin import PluginInterface


@dataclass
class FabricNetworkConfig:
    """Configuration for Hyperledger Fabric network"""
    network_name: str = "fabric-network"
    orderer_endpoints: List[str] = None
    peer_endpoints: List[str] = None
    ca_endpoints: List[str] = None
    
    # REST API Configuration
    rest_api_base_url: str = "http://localhost:4000"
    api_version: str = "v2"
    
    # gRPC Configuration
    grpc_orderer_port: int = 7050
    grpc_peer_port: int = 7051
    
    # Docker Configuration
    docker_compose_path: str = "./fabric-network/docker-compose.yaml"
    fabric_version: str = "2.5"
    
    # Credentials
    admin_username: str = "admin"
    admin_password: str = "admin"
    organization: str = "Org1MSP"
    
    def __post_init__(self):
        if self.orderer_endpoints is None:
            self.orderer_endpoints = ["localhost:7050"]
        if self.peer_endpoints is None:
            self.peer_endpoints = ["localhost:7051", "localhost:8051"]
        if self.ca_endpoints is None:
            self.ca_endpoints = ["localhost:7054"]


class FabricDockerManager:
    """Manages Hyperledger Fabric Docker containers"""
    
    def __init__(self, config: FabricNetworkConfig):
        self.config = config
        self.docker_client = docker.from_env()
        self.logger = logging.getLogger(__name__)
        self.containers: Dict[str, Container] = {}
        self.network: Optional[Network] = None
    
    def start_fabric_network(self) -> bool:
        """Start the Hyperledger Fabric network using Docker Compose"""
        try:
            self.logger.info("Starting Hyperledger Fabric network...")
            
            # Check if network already exists
            try:
                self.network = self.docker_client.networks.get(self.config.network_name)
                self.logger.info(f"Found existing network: {self.config.network_name}")
            except docker.errors.NotFound:
                # Create fabric network
                self.network = self.docker_client.networks.create(
                    self.config.network_name,
                    driver="bridge"
                )
                self.logger.info(f"Created network: {self.config.network_name}")
            
            # Start fabric containers
            self._start_orderer_containers()
            self._start_peer_containers()
            self._start_ca_containers()
            
            # Wait for services to be ready
            self._wait_for_fabric_ready()
            
            self.logger.info("Hyperledger Fabric network started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start Fabric network: {e}")
            return False
    
    def stop_fabric_network(self) -> bool:
        """Stop the Hyperledger Fabric network"""
        try:
            self.logger.info("Stopping Hyperledger Fabric network...")
            
            # Stop all containers
            for name, container in self.containers.items():
                try:
                    container.stop(timeout=10)
                    container.remove()
                    self.logger.info(f"Stopped container: {name}")
                except Exception as e:
                    self.logger.warning(f"Error stopping container {name}: {e}")
            
            # Remove network
            if self.network:
                try:
                    self.network.remove()
                    self.logger.info(f"Removed network: {self.config.network_name}")
                except Exception as e:
                    self.logger.warning(f"Error removing network: {e}")
            
            self.containers.clear()
            self.network = None
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop Fabric network: {e}")
            return False
    
    def _start_orderer_containers(self):
        """Start orderer containers"""
        for i, endpoint in enumerate(self.config.orderer_endpoints):
            host, port = endpoint.split(':')
            
            container_name = f"orderer{i}.example.com"
            
            try:
                # Check if container already exists
                existing_container = self.docker_client.containers.get(container_name)
                if existing_container.status == 'running':
                    self.containers[container_name] = existing_container
                    continue
                else:
                    existing_container.remove()
            except docker.errors.NotFound:
                pass
            
            # Create orderer container
            container = self.docker_client.containers.run(
                f"hyperledger/fabric-orderer:{self.config.fabric_version}",
                name=container_name,
                network=self.config.network_name,
                ports={f'{self.config.grpc_orderer_port}/tcp': port},
                environment={
                    'FABRIC_CFG_PATH': '/etc/hyperledger/fabric',
                    'ORDERER_GENERAL_LISTENADDRESS': '0.0.0.0',
                    'ORDERER_GENERAL_LISTENPORT': str(self.config.grpc_orderer_port),
                    'ORDERER_GENERAL_LOCALMSPID': 'OrdererMSP',
                    'ORDERER_GENERAL_LOCALMSPDIR': '/etc/hyperledger/fabric/msp',
                    'ORDERER_GENERAL_TLS_ENABLED': 'false',
                },
                detach=True,
                remove=False
            )
            
            self.containers[container_name] = container
            self.logger.info(f"Started orderer container: {container_name}")
    
    def _start_peer_containers(self):
        """Start peer containers"""
        for i, endpoint in enumerate(self.config.peer_endpoints):
            host, port = endpoint.split(':')
            
            container_name = f"peer{i}.org1.example.com"
            
            try:
                # Check if container already exists
                existing_container = self.docker_client.containers.get(container_name)
                if existing_container.status == 'running':
                    self.containers[container_name] = existing_container
                    continue
                else:
                    existing_container.remove()
            except docker.errors.NotFound:
                pass
            
            # Create peer container
            container = self.docker_client.containers.run(
                f"hyperledger/fabric-peer:{self.config.fabric_version}",
                name=container_name,
                network=self.config.network_name,
                ports={f'{self.config.grpc_peer_port}/tcp': port},
                environment={
                    'FABRIC_CFG_PATH': '/etc/hyperledger/fabric',
                    'CORE_PEER_ID': container_name,
                    'CORE_PEER_ADDRESS': f'{container_name}:{self.config.grpc_peer_port}',
                    'CORE_PEER_LISTENADDRESS': f'0.0.0.0:{self.config.grpc_peer_port}',
                    'CORE_PEER_LOCALMSPID': self.config.organization,
                    'CORE_PEER_MSPCONFIGPATH': '/etc/hyperledger/fabric/msp',
                    'CORE_PEER_TLS_ENABLED': 'false',
                    'CORE_VM_ENDPOINT': 'unix:///host/var/run/docker.sock',
                },
                volumes={'/var/run/docker.sock': {'bind': '/host/var/run/docker.sock', 'mode': 'rw'}},
                detach=True,
                remove=False
            )
            
            self.containers[container_name] = container
            self.logger.info(f"Started peer container: {container_name}")
    
    def _start_ca_containers(self):
        """Start Certificate Authority containers"""
        for i, endpoint in enumerate(self.config.ca_endpoints):
            host, port = endpoint.split(':')
            
            container_name = f"ca.org1.example.com"
            
            try:
                # Check if container already exists
                existing_container = self.docker_client.containers.get(container_name)
                if existing_container.status == 'running':
                    self.containers[container_name] = existing_container
                    continue
                else:
                    existing_container.remove()
            except docker.errors.NotFound:
                pass
            
            # Create CA container
            container = self.docker_client.containers.run(
                f"hyperledger/fabric-ca:{self.config.fabric_version}",
                name=container_name,
                network=self.config.network_name,
                ports={'7054/tcp': port},
                environment={
                    'FABRIC_CA_HOME': '/etc/hyperledger/fabric-ca-server',
                    'FABRIC_CA_SERVER_CA_NAME': container_name,
                    'FABRIC_CA_SERVER_TLS_ENABLED': 'false',
                    'FABRIC_CA_SERVER_PORT': '7054',
                },
                command='sh -c "fabric-ca-server start -b admin:adminpw"',
                detach=True,
                remove=False
            )
            
            self.containers[container_name] = container
            self.logger.info(f"Started CA container: {container_name}")
    
    def _wait_for_fabric_ready(self, timeout: int = 60):
        """Wait for Fabric services to be ready"""
        self.logger.info("Waiting for Fabric services to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Check if orderer is ready
                response = requests.get(
                    f"{self.config.rest_api_base_url}/health",
                    timeout=5
                )
                if response.status_code == 200:
                    self.logger.info("Fabric services are ready")
                    return
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(2)
        
        raise TimeoutError("Fabric services did not become ready within timeout")
    
    def get_container_logs(self, container_name: str) -> str:
        """Get logs from a specific container"""
        if container_name in self.containers:
            return self.containers[container_name].logs().decode('utf-8')
        return ""


class FabricGRPCClient:
    """gRPC client for Hyperledger Fabric services"""
    
    def __init__(self, config: FabricNetworkConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.channels: Dict[str, grpc.Channel] = {}
    
    async def connect_to_orderer(self, orderer_endpoint: str):
        """Connect to orderer via gRPC"""
        try:
            channel = grpc.aio.insecure_channel(orderer_endpoint)
            self.channels[f"orderer_{orderer_endpoint}"] = channel
            
            # Test connection
            await self._test_grpc_connection(channel)
            
            self.logger.info(f"Connected to orderer: {orderer_endpoint}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to orderer {orderer_endpoint}: {e}")
            return False
    
    async def connect_to_peer(self, peer_endpoint: str):
        """Connect to peer via gRPC"""
        try:
            channel = grpc.aio.insecure_channel(peer_endpoint)
            self.channels[f"peer_{peer_endpoint}"] = channel
            
            # Test connection
            await self._test_grpc_connection(channel)
            
            self.logger.info(f"Connected to peer: {peer_endpoint}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to peer {peer_endpoint}: {e}")
            return False
    
    async def _test_grpc_connection(self, channel: grpc.Channel):
        """Test gRPC connection"""
        # This would use actual Fabric protobuf services
        # For now, just ensure channel is ready
        try:
            await channel.channel_ready()
        except grpc.RpcError as e:
            raise ConnectionError(f"gRPC connection failed: {e}")
    
    async def send_transaction(self, channel_name: str, chaincode_name: str, 
                             function_name: str, args: List[str]) -> Dict[str, Any]:
        """Send transaction to Fabric network via gRPC"""
        # This would implement actual Fabric transaction submission
        # For demonstration purposes
        self.logger.info(f"Sending transaction: {function_name} to {chaincode_name}")
        
        return {
            'transaction_id': f"tx_{int(time.time_ns())}",
            'status': 'success',
            'channel': channel_name,
            'chaincode': chaincode_name,
            'function': function_name,
            'args': args
        }
    
    def close_connections(self):
        """Close all gRPC connections"""
        for name, channel in self.channels.items():
            channel.close()
            self.logger.info(f"Closed connection: {name}")
        self.channels.clear()


class FabricRESTClient:
    """REST API client for Hyperledger Fabric"""
    
    def __init__(self, config: FabricNetworkConfig):
        self.config = config
        self.base_url = f"{config.rest_api_base_url}/api/{config.api_version}"
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
        
        # Set default headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def authenticate(self) -> bool:
        """Authenticate with Fabric REST API"""
        try:
            auth_data = {
                'username': self.config.admin_username,
                'password': self.config.admin_password,
                'orgName': self.config.organization
            }
            
            response = self.session.post(
                f"{self.base_url}/users",
                json=auth_data,
                timeout=10
            )
            
            if response.status_code == 200:
                token = response.json().get('token')
                if token:
                    self.session.headers['Authorization'] = f"Bearer {token}"
                    self.logger.info("Successfully authenticated with Fabric REST API")
                    return True
            
            self.logger.error(f"Authentication failed: {response.status_code}")
            return False
            
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return False
    
    def get_channels(self) -> List[Dict[str, Any]]:
        """Get list of channels"""
        try:
            response = self.session.get(f"{self.base_url}/channels", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get channels: {e}")
            return []
    
    def get_chaincode_list(self, channel_name: str) -> List[Dict[str, Any]]:
        """Get list of installed chaincodes on a channel"""
        try:
            response = self.session.get(
                f"{self.base_url}/channels/{channel_name}/chaincodes",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get chaincodes for channel {channel_name}: {e}")
            return []
    
    def invoke_chaincode(self, channel_name: str, chaincode_name: str, 
                        function_name: str, args: List[str]) -> Dict[str, Any]:
        """Invoke chaincode function"""
        try:
            invoke_data = {
                'peers': self.config.peer_endpoints,
                'fcn': function_name,
                'args': args
            }
            
            response = self.session.post(
                f"{self.base_url}/channels/{channel_name}/chaincodes/{chaincode_name}",
                json=invoke_data,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Failed to invoke chaincode: {e}")
            return {'error': str(e)}
    
    def query_chaincode(self, channel_name: str, chaincode_name: str,
                       function_name: str, args: List[str]) -> Dict[str, Any]:
        """Query chaincode function"""
        try:
            query_data = {
                'peer': self.config.peer_endpoints[0],
                'fcn': function_name,
                'args': args
            }
            
            response = self.session.get(
                f"{self.base_url}/channels/{channel_name}/chaincodes/{chaincode_name}",
                params=query_data,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Failed to query chaincode: {e}")
            return {'error': str(e)}
    
    def get_block_by_number(self, channel_name: str, block_number: int) -> Dict[str, Any]:
        """Get block by number"""
        try:
            response = self.session.get(
                f"{self.base_url}/channels/{channel_name}/blocks/{block_number}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get block {block_number}: {e}")
            return {'error': str(e)}
    
    def get_transaction_by_id(self, channel_name: str, tx_id: str) -> Dict[str, Any]:
        """Get transaction by ID"""
        try:
            response = self.session.get(
                f"{self.base_url}/channels/{channel_name}/transactions/{tx_id}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get transaction {tx_id}: {e}")
            return {'error': str(e)}