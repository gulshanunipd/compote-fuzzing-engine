#!/usr/bin/env python3
"""
Basic COMPOTE Usage Example

Demonstrates how to use the COMPOTE fuzzing engine for consensus protocol testing.
This example shows:
1. Loading sample consensus messages
2. Initializing the fuzzing engine
3. Running a fuzzing campaign
4. Analyzing results
"""

import json
import time
import random
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from compote import CompoteFuzzer
from compote.core.types import RawMessage, MessageType


def generate_sample_messages(count: int = 50) -> list:
    """Generate sample consensus messages for demonstration"""
    messages = []
    
    message_types = ['propose', 'prevote', 'precommit', 'commit', 'round_change']
    node_ids = [f"node_{i}" for i in range(1, 11)]
    
    for i in range(count):
        # Generate realistic consensus message
        message_data = {
            'message_id': f"msg_{i}_{int(time.time_ns())}",
            'message_type': random.choice(message_types),
            'round_number': random.randint(1, 100),
            'view_number': random.randint(0, 10),
            'block_height': random.randint(1, 1000),
            'sender_id': random.choice(node_ids),
            'role': random.choice(['leader', 'validator', 'observer']),
            'timestamp': time.time() + random.uniform(-60, 60),
            'block_hash': f"hash_{random.randint(1000, 9999)}",
            'signature': f"sig_{random.randint(10000, 99999)}"
        }
        
        # Add type-specific fields
        if message_data['message_type'] == 'propose':
            message_data['proposal_hash'] = f"proposal_{random.randint(1000, 9999)}"
            message_data['block_data'] = f"block_data_{i}"
        elif message_data['message_type'] in ['prevote', 'precommit']:
            message_data['vote_type'] = random.choice(['yes', 'no', 'nil'])
        elif message_data['message_type'] == 'commit':
            message_data['commit_signatures'] = [f"commit_sig_{j}" for j in range(random.randint(2, 7))]
        elif message_data['message_type'] == 'round_change':
            message_data['new_round'] = message_data['round_number'] + 1
            message_data['justification'] = f"round_change_reason_{i}"
        
        # Convert to RawMessage
        raw_data = json.dumps(message_data).encode('utf-8')
        raw_message = RawMessage(
            data=raw_data,
            timestamp=message_data['timestamp'],
            source_id=message_data['sender_id']
        )
        
        messages.append(raw_message)
    
    return messages


def run_basic_fuzzing_example():
    """Run a basic fuzzing example"""
    print("🚀 COMPOTE Basic Fuzzing Example")
    print("=" * 50)
    
    # Configuration for the fuzzing engine
    config = {
        # Feature extraction
        'normalize_features': True,
        
        # Clustering
        'clustering_eps': 0.3,
        'clustering_min_samples': 3,
        
        # Priority calculation weights
        'priority_alpha': 0.3,  # similarity
        'priority_beta': 0.4,   # fault history
        'priority_gamma': 0.3,  # coverage
        
        # Mutation settings
        'timestamp_variance': 0.2,
        'round_variance': 3,
        'enabled_mutations': ['timestamp', 'round', 'view', 'payload'],
        
        # Execution settings
        'simulation_mode': True,  # Use simulation for demo
        'max_iterations': 100,
        'max_workers': 2
    }
    
    print("📋 Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Initialize COMPOTE fuzzer
    print("🔧 Initializing COMPOTE fuzzer...")
    with CompoteFuzzer(config) as fuzzer:
        
        # Generate and load sample messages
        print("📨 Generating sample consensus messages...")
        sample_messages = generate_sample_messages(30)
        loaded_count = fuzzer.load_messages(sample_messages, format_type='json')
        print(f"✅ Loaded {loaded_count} messages")
        
        # Initialize seed pool
        print("🌱 Initializing seed pool...")
        if not fuzzer.initialize_seed_pool():
            print("❌ Failed to initialize seed pool")
            return
        
        # Get initial statistics
        print("📊 Initial Statistics:")
        stats = fuzzer.get_comprehensive_report()
        print(f"  Messages processed: {stats['summary']['messages_processed']}")
        print(f"  Clusters created: {stats['summary']['clusters_created']}")
        print(f"  Active clusters: {stats['current_state']['active_clusters']}")
        print()
        
        # Set up progress callback
        def progress_callback(iteration, max_iterations, result):
            if iteration % 10 == 0:  # Print every 10 iterations
                progress = (iteration / max_iterations) * 100
                print(f"⚡ Progress: {progress:.1f}% (Iteration {iteration}/{max_iterations})")
                if result.fault_detected:
                    print(f"  🐛 Fault detected in message {result.message_id}")
                if result.new_paths_covered > 0:
                    print(f"  🔍 New paths covered: {result.new_paths_covered}")
        
        # Set up fault callback
        def fault_callback(result):
            print(f"🚨 FAULT DETECTED: {result.message_id}")
            print(f"   Error: {result.error_message}")
            print(f"   State changes: {result.state_changes}")
        
        fuzzer.set_progress_callback(progress_callback)
        fuzzer.set_fault_callback(fault_callback)
        
        # Start fuzzing
        print("🎯 Starting fuzzing campaign...")
        start_time = time.time()
        
        success = fuzzer.start_fuzzing(max_iterations=50)
        
        end_time = time.time()
        runtime = end_time - start_time
        
        if success:
            print(f"✅ Fuzzing completed in {runtime:.2f} seconds")
        else:
            print("❌ Fuzzing failed")
            return
        
        # Generate final report
        print("\n📈 Final Results:")
        print("=" * 30)
        
        final_stats = fuzzer.get_comprehensive_report()
        summary = final_stats['summary']
        performance = final_stats['performance']
        
        print(f"Total iterations: {summary['total_iterations']}")
        print(f"Runtime: {summary['total_runtime']:.2f} seconds")
        print(f"Faults discovered: {summary['faults_discovered']}")
        print(f"Execution errors: {summary['execution_errors']}")
        print(f"Iterations/second: {performance['iterations_per_second']:.2f}")
        print(f"Fault discovery rate: {performance['fault_discovery_rate']:.3f}")
        print(f"Success rate: {performance['success_rate']:.3f}")
        
        # Algorithm-specific statistics
        print("\n🔬 Algorithm Statistics:")
        algo_stats = final_stats['algorithm_stats']
        
        print("Clustering:")
        clustering_stats = algo_stats['clustering']['results']
        print(f"  Clusters: {clustering_stats['n_clusters']}")
        print(f"  Noise points: {clustering_stats['n_noise_points']}")
        print(f"  Silhouette score: {clustering_stats['silhouette_score']:.3f}")
        
        print("Priority Calculation:")
        priority_stats = algo_stats['priority_calculation']
        print(f"  Cache size: {priority_stats['cache_info']['cache_size']}")
        print(f"  Historical records: {priority_stats['historical_data']['records_count']}")
        
        print("State Analysis:")
        state_stats = algo_stats['state_analysis']['current_state']
        print(f"  Coverage: {state_stats['coverage_percentage']:.2f}%")
        print(f"  Fault rate: {algo_stats['state_analysis']['execution_stats']['fault_rate']:.3f}")
        
        # Save results
        output_file = "compote_example_results.json"
        fuzzer.save_state(output_file)
        print(f"\n💾 Results saved to {output_file}")


def run_clustering_example():
    """Demonstrate COMPOTE's clustering capabilities"""
    print("\n🎯 COMPOTE Clustering Example")
    print("=" * 40)
    
    # Generate messages with different characteristics
    messages = []
    
    # Group 1: Normal operation messages
    for i in range(15):
        msg_data = {
            'message_type': 'propose' if i % 3 == 0 else 'prevote',
            'round_number': random.randint(1, 5),
            'view_number': 0,
            'block_height': random.randint(100, 110),
            'sender_id': f"node_{i % 4}",
            'timestamp': time.time() + i
        }
        raw_data = json.dumps(msg_data).encode('utf-8')
        messages.append(RawMessage(raw_data, msg_data['timestamp'], msg_data['sender_id']))
    
    # Group 2: View change messages  
    for i in range(10):
        msg_data = {
            'message_type': 'round_change',
            'round_number': random.randint(10, 15),
            'view_number': random.randint(1, 3),
            'block_height': random.randint(200, 210),
            'sender_id': f"node_{i % 4}",
            'timestamp': time.time() + 100 + i
        }
        raw_data = json.dumps(msg_data).encode('utf-8')
        messages.append(RawMessage(raw_data, msg_data['timestamp'], msg_data['sender_id']))
    
    # Initialize fuzzer for clustering demo
    config = {'clustering_eps': 0.4, 'simulation_mode': True}
    
    with CompoteFuzzer(config) as fuzzer:
        loaded_count = fuzzer.load_messages(messages)
        print(f"📨 Loaded {loaded_count} messages")
        
        if fuzzer.initialize_seed_pool():
            clusters = fuzzer.current_clusters
            print(f"🎯 Created {len(clusters)} clusters:")
            
            for cluster_id, cluster in clusters.items():
                if cluster_id >= 0:  # Skip noise
                    print(f"  Cluster {cluster_id}: {len(cluster.message_ids)} messages")
                    
                    # Show some sample messages from cluster
                    sample_ids = cluster.message_ids[:3]
                    for msg_id in sample_ids:
                        if msg_id in fuzzer.parsed_messages:
                            msg = fuzzer.parsed_messages[msg_id]
                            print(f"    - {msg.message_type.value} R{msg.round_number} V{msg.view_number}")


if __name__ == "__main__":
    try:
        print("🎬 COMPOTE Fuzzing Engine Examples")
        print("Copyright (c) 2024 COMPOTE Implementation")
        print()
        
        # Run basic fuzzing example
        run_basic_fuzzing_example()
        
        # Run clustering example
        run_clustering_example()
        
        print("\n🎉 All examples completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()