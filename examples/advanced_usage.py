#!/usr/bin/env python3
"""
Advanced COMPOTE Usage Example with Plugin Integration

Demonstrates advanced features of COMPOTE including:
1. Plugin integration with LOKI/Tyr
2. Custom mutation strategies  
3. Real-time monitoring and analysis
4. Performance optimization
"""

import json
import time
import threading
import signal
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from compote import CompoteFuzzer
from compote.plugins import LokiPlugin, TyrPlugin
from compote.core.types import RawMessage
from compote.core.state_analyzer import MutationStrategy


class AdvancedFuzzingDemo:
    """Advanced fuzzing demonstration with plugin integration"""
    
    def __init__(self):
        self.fuzzer = None
        self.plugins = {}
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        self.results_log = []
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print("\n🛑 Received shutdown signal, cleaning up...")
        self.stop_monitoring.set()
        if self.fuzzer:
            self.fuzzer.stop_fuzzing()
        sys.exit(0)
    
    def setup_advanced_config(self) -> dict:
        """Setup advanced configuration for the fuzzing engine"""
        return {
            # Enhanced feature extraction
            'normalize_features': True,
            
            # Optimized clustering
            'clustering_eps': 0.3,
            'clustering_min_samples': 2,
            'scale_features': True,
            
            # Dynamic priority weights (will be adjusted during fuzzing)
            'priority_alpha': 0.25,  # similarity
            'priority_beta': 0.45,   # fault history  
            'priority_gamma': 0.30,  # coverage
            
            # Advanced thresholds
            'fault_threshold': 0.05,
            'similarity_threshold': 0.15,
            'time_threshold': 1800.0,  # 30 minutes
            
            # Enhanced mutation settings
            'timestamp_variance': 0.15,
            'round_variance': 5,
            'view_variance': 2,
            'enabled_mutations': ['timestamp', 'round', 'view', 'payload', 'sender', 'signature'],
            
            # Performance settings
            'max_workers': 4,
            'max_iterations': 500,
            'save_interval': 50,
            'auto_optimize': True,
            
            # Advanced execution
            'simulation_mode': True,  # Set to False for real testing
            'execution_timeout': 45.0,
            'max_retries': 5
        }
    
    def generate_complex_message_set(self, count: int = 100) -> list:
        """Generate a complex set of consensus messages with various scenarios"""
        messages = []
        
        # Scenario 1: Normal consensus flow (40%)
        for i in range(int(count * 0.4)):
            messages.extend(self._generate_normal_flow_messages(i))
        
        # Scenario 2: View change scenarios (25%)
        for i in range(int(count * 0.25)):
            messages.extend(self._generate_view_change_messages(i))
        
        # Scenario 3: Network partition recovery (20%)
        for i in range(int(count * 0.20)):
            messages.extend(self._generate_partition_recovery_messages(i))
        
        # Scenario 4: Byzantine behavior (15%)
        for i in range(int(count * 0.15)):
            messages.extend(self._generate_byzantine_messages(i))
        
        return messages
    
    def _generate_normal_flow_messages(self, base_id: int) -> list:
        """Generate normal consensus flow messages"""
        messages = []
        round_num = base_id % 50 + 1
        
        # Propose -> Prevote -> Precommit -> Commit sequence
        message_types = ['propose', 'prevote', 'precommit', 'commit']
        
        for i, msg_type in enumerate(message_types):
            msg_data = {
                'message_id': f"normal_{base_id}_{i}",
                'message_type': msg_type,
                'round_number': round_num,
                'view_number': 0,
                'block_height': round_num,
                'sender_id': f"node_{i % 4}",
                'role': 'validator' if i > 0 else 'leader',
                'timestamp': time.time() + base_id * 10 + i,
                'block_hash': f"block_hash_{round_num}",
                'signature': f"sig_{base_id}_{i}"
            }
            
            if msg_type == 'propose':
                msg_data['proposal_hash'] = f"proposal_{round_num}"
                msg_data['block_data'] = f"transactions_{round_num}"
            elif msg_type in ['prevote', 'precommit']:
                msg_data['vote_type'] = 'yes'
            
            raw_data = json.dumps(msg_data).encode('utf-8')
            messages.append(RawMessage(raw_data, msg_data['timestamp'], msg_data['sender_id']))
        
        return messages
    
    def _generate_view_change_messages(self, base_id: int) -> list:
        """Generate view change scenario messages"""
        messages = []
        
        msg_data = {
            'message_id': f"view_change_{base_id}",
            'message_type': 'round_change',
            'round_number': base_id % 30 + 10,
            'view_number': (base_id % 5) + 1,
            'block_height': base_id % 30 + 10,
            'sender_id': f"node_{base_id % 6}",
            'role': 'validator',
            'timestamp': time.time() + base_id * 15,
            'new_round': (base_id % 30) + 11,
            'justification': f"timeout_in_round_{base_id % 30 + 10}",
            'signature': f"vc_sig_{base_id}"
        }
        
        raw_data = json.dumps(msg_data).encode('utf-8')
        messages.append(RawMessage(raw_data, msg_data['timestamp'], msg_data['sender_id']))
        
        return messages
    
    def _generate_partition_recovery_messages(self, base_id: int) -> list:
        """Generate network partition recovery messages"""
        messages = []
        
        # Late messages from partitioned nodes
        msg_data = {
            'message_id': f"partition_{base_id}",
            'message_type': 'prevote',
            'round_number': base_id % 20 + 5,
            'view_number': 0,
            'block_height': base_id % 20 + 5,
            'sender_id': f"partitioned_node_{base_id % 3}",
            'role': 'validator',
            'timestamp': time.time() + base_id * 20 + 100,  # Late timestamp
            'vote_type': 'yes',
            'block_hash': f"old_block_hash_{base_id % 20 + 5}",
            'signature': f"part_sig_{base_id}"
        }
        
        raw_data = json.dumps(msg_data).encode('utf-8')
        messages.append(RawMessage(raw_data, msg_data['timestamp'], msg_data['sender_id']))
        
        return messages
    
    def _generate_byzantine_messages(self, base_id: int) -> list:
        """Generate Byzantine behavior messages"""
        messages = []
        
        # Double voting
        for vote_type in ['yes', 'no']:
            msg_data = {
                'message_id': f"byzantine_{base_id}_{vote_type}",
                'message_type': 'prevote',
                'round_number': base_id % 15 + 1,
                'view_number': 0,
                'block_height': base_id % 15 + 1,
                'sender_id': f"byzantine_node_{base_id % 2}",
                'role': 'validator',
                'timestamp': time.time() + base_id * 5,
                'vote_type': vote_type,
                'block_hash': f"conflicting_hash_{vote_type}_{base_id}",
                'signature': f"byz_sig_{base_id}_{vote_type}"
            }
            
            raw_data = json.dumps(msg_data).encode('utf-8')
            messages.append(RawMessage(raw_data, msg_data['timestamp'], msg_data['sender_id']))
        
        return messages
    
    def setup_plugins(self):
        """Setup and initialize plugins"""
        print("🔌 Setting up plugins...")
        
        # Setup LOKI plugin (if available)
        loki_config = {
            'loki_host': 'localhost',
            'loki_port': 9001,
            'seed_batch_size': 10,
            'auto_reseed': True,
            'reseed_threshold': 5
        }
        
        self.plugins['loki'] = LokiPlugin(loki_config)
        
        # Setup Tyr plugin (if available)
        tyr_config = {
            'tyr_path': '/usr/local/bin/tyr',  # Adjust path as needed
            'working_dir': '/tmp/compote_tyr_demo',
            'batch_size': 8,
            'monitoring_interval': 0.5
        }
        
        self.plugins['tyr'] = TyrPlugin(tyr_config)
        
        # Try to initialize plugins (they may fail if frameworks not available)
        for name, plugin in self.plugins.items():
            try:
                if plugin.initialize():
                    print(f"✅ {name.upper()} plugin initialized")
                    plugin.set_compote_engine(self.fuzzer)
                else:
                    print(f"⚠️ {name.upper()} plugin failed to initialize (framework may not be available)")
            except Exception as e:
                print(f"❌ {name.upper()} plugin error: {e}")
    
    def setup_monitoring(self):
        """Setup real-time monitoring"""
        print("📊 Setting up real-time monitoring...")
        
        def monitor():
            iteration_count = 0
            last_report_time = time.time()
            
            while not self.stop_monitoring.is_set():
                try:
                    if self.fuzzer and self.fuzzer.is_running:
                        # Get current statistics
                        stats = self.fuzzer.get_comprehensive_report()
                        current_time = time.time()
                        
                        # Report every 30 seconds
                        if current_time - last_report_time > 30:
                            self._print_monitoring_report(stats)
                            last_report_time = current_time
                        
                        # Check for significant events
                        if stats['summary']['faults_discovered'] > len([r for r in self.results_log if r.get('fault_detected')]):
                            print(f"🚨 New fault discovered! Total: {stats['summary']['faults_discovered']}")
                    
                    time.sleep(5)  # Monitor every 5 seconds
                    
                except Exception as e:
                    print(f"Monitoring error: {e}")
                    time.sleep(5)
        
        self.monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self.monitoring_thread.start()
    
    def _print_monitoring_report(self, stats):
        """Print real-time monitoring report"""
        print("\n" + "="*60)
        print("📈 REAL-TIME MONITORING REPORT")
        print("="*60)
        
        summary = stats['summary']
        performance = stats['performance']
        
        print(f"Runtime: {summary['total_runtime']:.1f}s | "
              f"Iterations: {summary['total_iterations']} | "
              f"Rate: {performance['iterations_per_second']:.2f}/s")
        
        print(f"Faults: {summary['faults_discovered']} | "
              f"Errors: {summary['execution_errors']} | "
              f"Success: {performance['success_rate']:.2%}")
        
        # Algorithm performance
        algo_stats = stats['algorithm_stats']
        clustering = algo_stats['clustering']['results']
        
        print(f"Clusters: {clustering['n_clusters']} | "
              f"Silhouette: {clustering['silhouette_score']:.3f}")
        
        # Plugin status
        for name, plugin in self.plugins.items():
            if plugin.is_active:
                plugin_stats = plugin.get_statistics()
                print(f"{name.upper()}: {plugin_stats['statistics']['messages_provided']} provided, "
                      f"{plugin_stats['statistics']['faults_found']} faults")
        
        print("="*60)
    
    def run_advanced_fuzzing_campaign(self):
        """Run the advanced fuzzing campaign"""
        print("🚀 Starting Advanced COMPOTE Fuzzing Campaign")
        print("="*60)
        
        # Setup configuration
        config = self.setup_advanced_config()
        
        # Initialize fuzzer
        self.fuzzer = CompoteFuzzer(config)
        
        try:
            # Generate complex message set
            print("📝 Generating complex message scenarios...")
            messages = self.generate_complex_message_set(80)
            
            # Load messages
            loaded_count = self.fuzzer.load_messages(messages, format_type='json')
            print(f"📨 Loaded {loaded_count} messages")
            
            # Initialize seed pool
            print("🌱 Initializing advanced seed pool...")
            if not self.fuzzer.initialize_seed_pool():
                print("❌ Failed to initialize seed pool")
                return
            
            # Setup plugins
            self.setup_plugins()
            
            # Setup callbacks
            self._setup_advanced_callbacks()
            
            # Setup monitoring
            self.setup_monitoring()
            
            # Print initial statistics
            print("\n📊 Initial Configuration:")
            initial_stats = self.fuzzer.get_comprehensive_report()
            print(f"  Messages: {initial_stats['summary']['messages_processed']}")
            print(f"  Clusters: {initial_stats['summary']['clusters_created']}")
            print(f"  Features: {len(self.fuzzer.feature_extractor.feature_names)}")
            
            # Start fuzzing with plugins
            print("\n🎯 Starting advanced fuzzing campaign...")
            
            # Plugin-assisted fuzzing
            if any(plugin.is_active for plugin in self.plugins.values()):
                self._run_plugin_assisted_fuzzing()
            else:
                # Standalone fuzzing
                self.fuzzer.start_fuzzing(max_iterations=200)
            
            # Generate comprehensive report
            self._generate_final_report()
            
        except Exception as e:
            print(f"❌ Advanced fuzzing failed: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            self._cleanup()
    
    def _setup_advanced_callbacks(self):
        """Setup advanced callback functions"""
        
        def advanced_progress_callback(iteration, max_iterations, result):
            # Store result for analysis
            result_data = {
                'iteration': iteration,
                'timestamp': time.time(),
                'success': result.success,
                'fault_detected': result.fault_detected,
                'new_paths': result.new_paths_covered,
                'execution_time': result.execution_time
            }
            self.results_log.append(result_data)
            
            # Print significant events
            if result.fault_detected:
                print(f"🐛 Iteration {iteration}: Fault in {result.message_id}")
            elif result.new_paths_covered > 3:
                print(f"🔍 Iteration {iteration}: {result.new_paths_covered} new paths discovered")
        
        def advanced_fault_callback(result):
            print(f"\n🚨 CRITICAL FAULT DETECTED")
            print(f"   Message ID: {result.message_id}")
            print(f"   Error: {result.error_message}")
            print(f"   State Changes: {result.state_changes}")
            print(f"   Execution Time: {result.execution_time:.3f}s")
            
            # Log to file for analysis
            fault_data = {
                'timestamp': time.time(),
                'message_id': result.message_id,
                'error_message': result.error_message,
                'state_changes': result.state_changes,
                'coverage_metrics': result.coverage_metrics
            }
            
            with open('compote_faults.jsonl', 'a') as f:
                f.write(json.dumps(fault_data) + '\n')
        
        self.fuzzer.set_progress_callback(advanced_progress_callback)
        self.fuzzer.set_fault_callback(advanced_fault_callback)
    
    def _run_plugin_assisted_fuzzing(self):
        """Run fuzzing with plugin assistance"""
        print("🔗 Running plugin-assisted fuzzing...")
        
        active_plugins = [p for p in self.plugins.values() if p.is_active]
        
        for iteration in range(100):
            if self.stop_monitoring.is_set():
                break
            
            # Provide seeds to active plugins
            for plugin in active_plugins:
                try:
                    seeds = plugin.provide_seed_messages(5)
                    if seeds:
                        print(f"📤 Provided {len(seeds)} seeds to {plugin.name}")
                except Exception as e:
                    print(f"Plugin error: {e}")
            
            # Run normal fuzzing iteration
            if self.fuzzer.current_clusters:
                result = self.fuzzer.state_analyzer.run_fuzzing_iteration(
                    self.fuzzer.current_clusters,
                    self.fuzzer.priority_calculator
                )
                
                # Share results with plugins
                for plugin in active_plugins:
                    try:
                        plugin.receive_feedback(result)
                    except Exception as e:
                        print(f"Plugin feedback error: {e}")
            
            time.sleep(0.1)  # Small delay between iterations
    
    def _generate_final_report(self):
        """Generate comprehensive final report"""
        print("\n" + "="*80)
        print("📈 ADVANCED FUZZING CAMPAIGN - FINAL REPORT")
        print("="*80)
        
        final_stats = self.fuzzer.get_comprehensive_report()
        
        # Summary
        summary = final_stats['summary']
        print(f"📊 CAMPAIGN SUMMARY:")
        print(f"   Total Runtime: {summary['total_runtime']:.2f} seconds")
        print(f"   Total Iterations: {summary['total_iterations']}")
        print(f"   Messages Processed: {summary['messages_processed']}")
        print(f"   Faults Discovered: {summary['faults_discovered']}")
        print(f"   Execution Errors: {summary['execution_errors']}")
        
        # Performance
        performance = final_stats['performance']
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"   Iterations/Second: {performance['iterations_per_second']:.2f}")
        print(f"   Fault Discovery Rate: {performance['fault_discovery_rate']:.4f}")
        print(f"   Success Rate: {performance['success_rate']:.2%}")
        
        # Algorithm Analysis
        algo_stats = final_stats['algorithm_stats']
        print(f"\n🔬 ALGORITHM ANALYSIS:")
        
        # Clustering
        clustering = algo_stats['clustering']['results']
        print(f"   Clustering Quality:")
        print(f"     - Clusters: {clustering['n_clusters']}")
        print(f"     - Silhouette Score: {clustering['silhouette_score']:.3f}")
        print(f"     - Noise Points: {clustering['n_noise_points']}")
        
        # Priority Calculation
        priority_stats = algo_stats['priority_calculation']
        print(f"   Priority Calculation:")
        print(f"     - Cache Hits: {priority_stats['statistics']['cache_hits']}")
        print(f"     - Total Calculations: {priority_stats['statistics']['total_calculations']}")
        
        # Plugin Performance
        print(f"\n🔌 PLUGIN PERFORMANCE:")
        for name, plugin in self.plugins.items():
            if plugin.is_active:
                stats = plugin.get_detailed_statistics()
                print(f"   {name.upper()}:")
                print(f"     - Messages Provided: {stats['statistics']['messages_provided']}")
                print(f"     - Messages Executed: {stats['statistics']['messages_executed']}")
                print(f"     - Faults Found: {stats['statistics']['faults_found']}")
        
        # Save detailed results
        with open('advanced_fuzzing_report.json', 'w') as f:
            json.dump(final_stats, f, indent=2, default=str)
        
        print(f"\n💾 Detailed report saved to 'advanced_fuzzing_report.json'")
        print("="*80)
    
    def _cleanup(self):
        """Cleanup resources"""
        print("\n🧹 Cleaning up...")
        
        # Stop monitoring
        self.stop_monitoring.set()
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        # Shutdown plugins
        for name, plugin in self.plugins.items():
            if plugin.is_active:
                try:
                    plugin.shutdown()
                    print(f"✅ {name.upper()} plugin shutdown")
                except Exception as e:
                    print(f"⚠️ Error shutting down {name.upper()}: {e}")
        
        # Stop fuzzer
        if self.fuzzer and self.fuzzer.is_running:
            self.fuzzer.stop_fuzzing()
        
        print("✅ Cleanup complete")


def main():
    """Main function for advanced demo"""
    print("🎬 COMPOTE Advanced Fuzzing Demo")
    print("Copyright (c) 2024 COMPOTE Implementation")
    print()
    
    demo = AdvancedFuzzingDemo()
    
    try:
        demo.run_advanced_fuzzing_campaign()
        print("\n🎉 Advanced fuzzing campaign completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Campaign interrupted by user")
    except Exception as e:
        print(f"\n❌ Campaign failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()