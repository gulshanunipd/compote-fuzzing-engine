#!/usr/bin/env python3
"""
COMPOTE Hyperledger Fabric Integration Example

Demonstrates how to use COMPOTE with Hyperledger Fabric v2.5:
1. Starting a Fabric network using Docker
2. Extracting consensus messages from Fabric logs
3. Fuzzing the Fabric network with mutated consensus messages
4. Monitoring blockchain state and chaincode execution
5. Collecting coverage metrics from Fabric components
"""

import sys
import time
import asyncio
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from compote import CompoteFuzzer
from compote.fabric.fabric_plugin import FabricPlugin
from compote.fabric.fabric_integration import FabricNetworkConfig


class FabricFuzzingDemo:
    """Demonstration of COMPOTE fuzzing with Hyperledger Fabric"""
    
    def __init__(self):
        # Fabric configuration
        self.fabric_config = {
            'network_name': 'compote-fabric-test',
            'orderer_endpoints': ['localhost:7050'],
            'peer_endpoints': ['localhost:7051', 'localhost:8051'],
            'rest_api_base_url': 'http://localhost:4000',
            'docker_compose_path': './fabric-network/docker-compose.yaml',
            'fabric_version': '2.5',
            'organization': 'Org1MSP',
            'admin_username': 'admin',
            'admin_password': 'admin'
        }
        
        # COMPOTE configuration optimized for Fabric
        self.compote_config = {
            # Enable real execution (not simulation)
            'simulation_mode': False,
            
            # Fabric-specific clustering
            'clustering_eps': 0.4,
            'clustering_min_samples': 2,
            
            # Priority weights for blockchain consensus
            'priority_alpha': 0.2,  # similarity weight (lower for diverse testing)
            'priority_beta': 0.5,   # fault weight (higher for security focus)
            'priority_gamma': 0.3,  # coverage weight
            
            # Execution settings
            'max_iterations': 50,
            'execution_timeout': 60.0,  # Longer timeout for blockchain operations
            'max_retries': 3,
            
            # Performance
            'max_workers': 2,  # Moderate concurrency for Fabric
            'save_interval': 10,
            'auto_optimize': True
        }
        
        self.fabric_plugin = None
        self.fuzzer = None
    
    def setup_fabric_network(self) -> bool:
        """Setup and start Hyperledger Fabric network"""
        print("🔧 Setting up Hyperledger Fabric v2.5 network...")
        print("=" * 60)
        
        try:
            # Initialize Fabric plugin
            self.fabric_plugin = FabricPlugin(self.fabric_config)
            
            # Start Fabric network
            if not self.fabric_plugin.initialize():
                print("❌ Failed to initialize Fabric network")
                return False
            
            print("✅ Fabric network started successfully")
            
            # Display network status
            status = self.fabric_plugin.get_fabric_network_status()
            print(f"📊 Network Status:")
            print(f"  Containers: {len(status.get('containers', {}))}")
            print(f"  Channels: {status.get('channels', [])}")
            print(f"  Running: {status.get('network_running', False)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error setting up Fabric network: {e}")
            return False
    
    def run_fabric_consensus_fuzzing(self):
        """Run COMPOTE fuzzing on Fabric consensus layer"""
        print("\n🎯 Starting Fabric Consensus Fuzzing")
        print("=" * 50)
        
        try:
            # Initialize COMPOTE with Fabric plugin
            self.fuzzer = CompoteFuzzer(self.compote_config)
            self.fabric_plugin.set_compote_engine(self.fuzzer)
            
            # Extract initial consensus messages from Fabric
            print("📨 Extracting consensus messages from Fabric network...")
            seed_messages = self.fabric_plugin.provide_seed_messages(20)
            
            if not seed_messages:
                print("⚠️ No consensus messages extracted, generating synthetic messages...")
                seed_messages = self._generate_fabric_synthetic_messages()
            
            print(f"✅ Extracted {len(seed_messages)} seed messages")
            
            # Load messages into COMPOTE
            loaded_count = self.fuzzer.load_messages([
                {'message_type': msg.message_type.value,
                 'round_number': msg.round_number,
                 'view_number': msg.view_number,
                 'sender_id': msg.sender_id,
                 'timestamp': msg.timestamp}
                for msg in seed_messages
            ])
            
            print(f"📥 Loaded {loaded_count} messages into COMPOTE")
            
            # Initialize seed pool
            if not self.fuzzer.initialize_seed_pool():
                print("❌ Failed to initialize COMPOTE seed pool")
                return
            
            print("🌱 COMPOTE seed pool initialized")
            print(f"  Clusters: {len(self.fuzzer.current_clusters)}")
            print(f"  Features: {len(self.fuzzer.message_features)}")
            
            # Setup fuzzing callbacks
            self._setup_fabric_callbacks()
            
            # Run fuzzing campaign
            print("\n🚀 Starting Fabric consensus fuzzing campaign...")
            start_time = time.time()
            
            success = self.fuzzer.start_fuzzing(max_iterations=25)
            
            end_time = time.time()
            
            if success:
                print(f"✅ Fuzzing completed in {end_time - start_time:.2f} seconds")
                self._analyze_fabric_results()
            else:
                print("❌ Fuzzing campaign failed")
            
        except Exception as e:
            print(f"❌ Error during Fabric fuzzing: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_fabric_synthetic_messages(self):
        """Generate synthetic Fabric consensus messages for testing"""
        from compote.core.types import ParsedMessage, MessageType, NodeRole
        
        messages = []
        
        # Generate typical Fabric consensus sequence
        for round_num in range(1, 6):
            # Block proposal
            messages.append(ParsedMessage(
                message_id=f"fabric_propose_{round_num}",
                message_type=MessageType.PROPOSE,
                round_number=round_num,
                view_number=0,
                block_height=round_num,
                sender_id="orderer.example.com",
                sender_role=NodeRole.LEADER,
                timestamp=time.time() + round_num,
                payload_hash=f"fabric_block_{round_num}",
                additional_fields={
                    'fabric_source': 'synthetic',
                    'block_transactions': random.randint(1, 10)
                }
            ))
            
            # Peer validations
            for peer_id in ['peer0.org1.example.com', 'peer1.org1.example.com']:
                messages.append(ParsedMessage(
                    message_id=f"fabric_commit_{round_num}_{peer_id}",
                    message_type=MessageType.COMMIT,
                    round_number=round_num,
                    view_number=0,
                    block_height=round_num,
                    sender_id=peer_id,
                    sender_role=NodeRole.VALIDATOR,
                    timestamp=time.time() + round_num + 0.5,
                    payload_hash=f"fabric_commit_{round_num}",
                    additional_fields={
                        'fabric_source': 'synthetic',
                        'endorsement': True
                    }
                ))
        
        return messages
    
    def _setup_fabric_callbacks(self):
        """Setup callbacks for Fabric-specific monitoring"""
        
        def fabric_progress_callback(iteration, max_iterations, result):
            if iteration % 5 == 0:
                progress = (iteration / max_iterations) * 100
                print(f"⚡ Fuzzing Progress: {progress:.1f}% (Iteration {iteration}/{max_iterations})")
                
                if result.fault_detected:
                    print(f"  🔥 Fabric fault detected: {result.error_message}")
                
                if result.state_changes:
                    print(f"  📝 State changes: {', '.join(result.state_changes)}")
        
        def fabric_fault_callback(result):
            print(f"\n🚨 FABRIC FAULT DETECTED!")
            print(f"   Message ID: {result.message_id}")
            print(f"   Error: {result.error_message}")
            print(f"   State Changes: {result.state_changes}")
            print(f"   Coverage Impact: {result.coverage_metrics}")
            
            # Log to Fabric-specific fault file
            with open('fabric_faults.log', 'a') as f:
                f.write(f"{time.time()}: {result.message_id} - {result.error_message}\n")
        
        self.fuzzer.set_progress_callback(fabric_progress_callback)
        self.fuzzer.set_fault_callback(fabric_fault_callback)
    
    def _analyze_fabric_results(self):
        """Analyze Fabric fuzzing results"""
        print("\n📊 Fabric Fuzzing Analysis")
        print("=" * 40)
        
        # Get COMPOTE results
        report = self.fuzzer.get_comprehensive_report()
        summary = report['summary']
        
        print(f"🎯 Campaign Summary:")
        print(f"  Total Iterations: {summary['total_iterations']}")
        print(f"  Messages Processed: {summary['messages_processed']}")
        print(f"  Faults Discovered: {summary['faults_discovered']}")
        print(f"  Runtime: {summary['total_runtime']:.2f} seconds")
        
        # Get Fabric-specific results
        fabric_stats = self.fabric_plugin.get_detailed_statistics()
        
        print(f"\n🔗 Fabric Integration Stats:")
        print(f"  Consensus Messages Extracted: {fabric_stats['consensus_messages_extracted']}")
        print(f"  Active Transactions: {fabric_stats['active_transactions']}")
        print(f"  Coverage Data Points: {fabric_stats['coverage_data_points']}")
        
        # Network status
        network_status = fabric_stats['network_status']
        print(f"\n🌐 Fabric Network Status:")
        print(f"  Network Running: {network_status.get('network_running', False)}")
        print(f"  Active Containers: {len(network_status.get('containers', {}))}")
        print(f"  Channels: {network_status.get('channels', [])}")
        
        # Performance analysis
        performance = report['performance']
        print(f"\n⚡ Performance Metrics:")
        print(f"  Iterations/Second: {performance['iterations_per_second']:.2f}")
        print(f"  Fault Discovery Rate: {performance['fault_discovery_rate']:.4f}")
        print(f"  Success Rate: {performance['success_rate']:.2%}")
        
        # Save detailed report
        self.fuzzer.save_state('fabric_fuzzing_results.json')
        print(f"\n💾 Detailed results saved to 'fabric_fuzzing_results.json'")
    
    def cleanup(self):
        """Cleanup Fabric network and COMPOTE resources"""
        print("\n🧹 Cleaning up resources...")
        
        if self.fuzzer and self.fuzzer.is_running:
            self.fuzzer.stop_fuzzing()
        
        if self.fabric_plugin and self.fabric_plugin.is_active:
            self.fabric_plugin.shutdown()
        
        print("✅ Cleanup completed")
    
    def run_complete_demo(self):
        """Run the complete Fabric integration demo"""
        print("🎬 COMPOTE + Hyperledger Fabric v2.5 Integration Demo")
        print("=" * 70)
        print("This demo shows COMPOTE fuzzing a real Hyperledger Fabric network")
        print("using gRPC, REST API, and Docker integration.\n")
        
        try:
            # Setup Fabric network
            if not self.setup_fabric_network():
                return
            
            # Wait for network stabilization
            print("⏳ Waiting for Fabric network to stabilize...")
            time.sleep(10)
            
            # Run fuzzing
            self.run_fabric_consensus_fuzzing()
            
            print("\n🎉 Fabric integration demo completed successfully!")
            
        except KeyboardInterrupt:
            print("\n⏹️ Demo interrupted by user")
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()


def main():
    """Main function for Fabric integration demo"""
    demo = FabricFuzzingDemo()
    demo.run_complete_demo()


if __name__ == "__main__":
    # Import here to avoid issues if not all dependencies are installed
    import random
    main()