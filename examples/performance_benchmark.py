#!/usr/bin/env python3
"""
COMPOTE Performance Benchmarking Script

Benchmarks the performance of different COMPOTE algorithms and configurations.
Useful for:
1. Performance tuning
2. Comparative analysis
3. Scalability testing
4. Algorithm optimization
"""

import time
import json
import statistics
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from compote import CompoteFuzzer
from compote.core.types import RawMessage
from compote.core.parser import MessageParser
from compote.core.feature_extractor import FeatureExtractor
from compote.core.clustering import ContextClustering
from compote.core.priority_calculator import PriorityCalculator


class CompotePerformanceBenchmark:
    """Performance benchmarking suite for COMPOTE algorithms"""
    
    def __init__(self):
        self.results = {}
        self.test_data_sizes = [10, 25, 50, 100, 200, 500]
        self.test_iterations = 5
    
    def generate_test_data(self, size: int) -> list:
        """Generate test data of specified size"""
        messages = []
        
        for i in range(size):
            msg_data = {
                'message_id': f"bench_msg_{i}",
                'message_type': ['propose', 'prevote', 'precommit', 'commit'][i % 4],
                'round_number': i // 10 + 1,
                'view_number': i // 50,
                'block_height': i // 10 + 1,
                'sender_id': f"node_{i % 20}",
                'role': 'validator',
                'timestamp': time.time() + i,
                'block_hash': f"hash_{i // 10}",
                'signature': f"sig_{i}"
            }
            
            raw_data = json.dumps(msg_data).encode('utf-8')
            messages.append(RawMessage(raw_data, msg_data['timestamp'], msg_data['sender_id']))
        
        return messages
    
    def benchmark_parsing(self):
        """Benchmark message parsing performance"""
        print("🔍 Benchmarking message parsing...")
        
        parser = MessageParser()
        results = {'sizes': [], 'times': [], 'throughput': []}
        
        for size in self.test_data_sizes:
            print(f"  Testing with {size} messages...")
            
            # Generate test messages
            test_messages = self.generate_test_data(size)
            
            # Run multiple iterations
            iteration_times = []
            for iteration in range(self.test_iterations):
                start_time = time.perf_counter()
                
                for raw_msg in test_messages:
                    try:
                        parsed_msg = parser.parse(raw_msg, format_type='json')
                    except Exception:
                        pass  # Ignore parsing errors for benchmarking
                
                end_time = time.perf_counter()
                iteration_times.append(end_time - start_time)
            
            # Calculate statistics
            avg_time = statistics.mean(iteration_times)
            throughput = size / avg_time
            
            results['sizes'].append(size)
            results['times'].append(avg_time)
            results['throughput'].append(throughput)
            
            print(f"    Average time: {avg_time:.3f}s, Throughput: {throughput:.1f} msg/s")
        
        self.results['parsing'] = results
    
    def benchmark_feature_extraction(self):
        """Benchmark feature extraction performance"""
        print("🔍 Benchmarking feature extraction...")
        
        parser = MessageParser()
        extractor = FeatureExtractor(normalize=True)
        results = {'sizes': [], 'times': [], 'throughput': []}
        
        for size in self.test_data_sizes:
            print(f"  Testing with {size} messages...")
            
            # Prepare parsed messages
            test_messages = self.generate_test_data(size)
            parsed_messages = []
            
            for raw_msg in test_messages:
                try:
                    parsed_msg = parser.parse(raw_msg, format_type='json')
                    parsed_messages.append(parsed_msg)
                except Exception:
                    continue
            
            if not parsed_messages:
                continue
            
            # Run feature extraction iterations
            iteration_times = []
            for iteration in range(self.test_iterations):
                start_time = time.perf_counter()
                
                features = extractor.extract_features(parsed_messages)
                
                end_time = time.perf_counter()
                iteration_times.append(end_time - start_time)
            
            # Calculate statistics
            avg_time = statistics.mean(iteration_times)
            throughput = len(parsed_messages) / avg_time
            
            results['sizes'].append(len(parsed_messages))
            results['times'].append(avg_time)
            results['throughput'].append(throughput)
            
            print(f"    Average time: {avg_time:.3f}s, Throughput: {throughput:.1f} msg/s")
        
        self.results['feature_extraction'] = results
    
    def benchmark_clustering(self):
        """Benchmark clustering performance"""
        print("🔍 Benchmarking clustering...")
        
        parser = MessageParser()
        extractor = FeatureExtractor(normalize=True)
        clustering = ContextClustering(eps=0.3, min_samples=2)
        results = {'sizes': [], 'times': [], 'throughput': []}
        
        for size in self.test_data_sizes:
            print(f"  Testing with {size} messages...")
            
            # Prepare feature data
            test_messages = self.generate_test_data(size)
            parsed_messages = []
            
            for raw_msg in test_messages:
                try:
                    parsed_msg = parser.parse(raw_msg, format_type='json')
                    parsed_messages.append(parsed_msg)
                except Exception:
                    continue
            
            if len(parsed_messages) < 5:  # Need minimum messages for clustering
                continue
            
            features = extractor.extract_features(parsed_messages)
            
            # Run clustering iterations
            iteration_times = []
            for iteration in range(self.test_iterations):
                start_time = time.perf_counter()
                
                clusters = clustering.fit_predict(features)
                
                end_time = time.perf_counter()
                iteration_times.append(end_time - start_time)
            
            # Calculate statistics
            avg_time = statistics.mean(iteration_times)
            throughput = len(features) / avg_time
            
            results['sizes'].append(len(features))
            results['times'].append(avg_time)
            results['throughput'].append(throughput)
            
            print(f"    Average time: {avg_time:.3f}s, Throughput: {throughput:.1f} msg/s")
        
        self.results['clustering'] = results
    
    def benchmark_priority_calculation(self):
        """Benchmark priority calculation performance"""
        print("🔍 Benchmarking priority calculation...")
        
        parser = MessageParser()
        extractor = FeatureExtractor(normalize=True)
        clustering = ContextClustering(eps=0.3, min_samples=2)
        priority_calc = PriorityCalculator()
        results = {'sizes': [], 'times': [], 'throughput': []}
        
        for size in self.test_data_sizes:
            print(f"  Testing with {size} messages...")
            
            # Prepare clustered data
            test_messages = self.generate_test_data(size)
            parsed_messages = []
            
            for raw_msg in test_messages:
                try:
                    parsed_msg = parser.parse(raw_msg, format_type='json')
                    parsed_messages.append(parsed_msg)
                except Exception:
                    continue
            
            if len(parsed_messages) < 5:
                continue
            
            features = extractor.extract_features(parsed_messages)
            clusters = clustering.fit_predict(features)
            
            # Create feature mapping
            message_features = {f.message_id: f for f in features}
            
            # Run priority calculation iterations
            iteration_times = []
            for iteration in range(self.test_iterations):
                start_time = time.perf_counter()
                
                priorities = priority_calc.calculate_priorities(clusters, message_features)
                
                end_time = time.perf_counter()
                iteration_times.append(end_time - start_time)
            
            # Calculate statistics
            avg_time = statistics.mean(iteration_times)
            throughput = len(message_features) / avg_time
            
            results['sizes'].append(len(message_features))
            results['times'].append(avg_time)
            results['throughput'].append(throughput)
            
            print(f"    Average time: {avg_time:.3f}s, Throughput: {throughput:.1f} msg/s")
        
        self.results['priority_calculation'] = results
    
    def benchmark_end_to_end(self):
        """Benchmark end-to-end fuzzing performance"""
        print("🔍 Benchmarking end-to-end performance...")
        
        results = {'sizes': [], 'times': [], 'throughput': [], 'iterations': []}
        
        # Use smaller sizes for end-to-end testing
        e2e_sizes = [10, 25, 50, 100]
        
        for size in e2e_sizes:
            print(f"  Testing with {size} messages...")
            
            # Generate test data
            test_messages = self.generate_test_data(size)
            
            # Run end-to-end fuzzing
            iteration_times = []
            iterations_completed = []
            
            for test_run in range(3):  # Fewer iterations for E2E
                config = {
                    'simulation_mode': True,
                    'max_iterations': 20,  # Limited iterations for benchmark
                    'max_workers': 1  # Single threaded for consistent timing
                }
                
                with CompoteFuzzer(config) as fuzzer:
                    # Load and initialize
                    loaded_count = fuzzer.load_messages(test_messages)
                    if loaded_count == 0:
                        continue
                    
                    if not fuzzer.initialize_seed_pool():
                        continue
                    
                    # Time the fuzzing process
                    start_time = time.perf_counter()
                    
                    success = fuzzer.start_fuzzing(max_iterations=20)
                    
                    end_time = time.perf_counter()
                    
                    if success:
                        total_time = end_time - start_time
                        iteration_times.append(total_time)
                        
                        stats = fuzzer.get_comprehensive_report()
                        iterations_completed.append(stats['summary']['total_iterations'])
            
            if iteration_times:
                avg_time = statistics.mean(iteration_times)
                avg_iterations = statistics.mean(iterations_completed)
                throughput = avg_iterations / avg_time
                
                results['sizes'].append(size)
                results['times'].append(avg_time)
                results['throughput'].append(throughput)
                results['iterations'].append(avg_iterations)
                
                print(f"    Average time: {avg_time:.3f}s, Iterations/s: {throughput:.2f}")
        
        self.results['end_to_end'] = results
    
    def run_all_benchmarks(self):
        """Run all performance benchmarks"""
        print("🚀 Starting COMPOTE Performance Benchmarks")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run individual algorithm benchmarks
        self.benchmark_parsing()
        print()
        
        self.benchmark_feature_extraction()
        print()
        
        self.benchmark_clustering()
        print()
        
        self.benchmark_priority_calculation()
        print()
        
        self.benchmark_end_to_end()
        print()
        
        total_time = time.time() - start_time
        print(f"⏱️ Total benchmark time: {total_time:.2f} seconds")
        
        # Generate report
        self.generate_performance_report()
        
        # Create visualizations
        self.create_performance_plots()
    
    def generate_performance_report(self):
        """Generate detailed performance report"""
        print("📊 Performance Analysis Report")
        print("=" * 50)
        
        for algorithm, data in self.results.items():
            if not data['sizes']:
                continue
            
            print(f"\n{algorithm.upper()} PERFORMANCE:")
            print(f"  Input sizes tested: {data['sizes']}")
            
            if data['times']:
                print(f"  Execution times: {[f'{t:.3f}s' for t in data['times']]}")
                print(f"  Average time: {statistics.mean(data['times']):.3f}s")
                print(f"  Time std dev: {statistics.stdev(data['times']) if len(data['times']) > 1 else 0:.3f}s")
            
            if data['throughput']:
                print(f"  Throughput: {[f'{t:.1f}' for t in data['throughput']]}")
                print(f"  Average throughput: {statistics.mean(data['throughput']):.1f} ops/s")
                print(f"  Peak throughput: {max(data['throughput']):.1f} ops/s")
            
            # Calculate scalability (time complexity)
            if len(data['sizes']) > 1 and len(data['times']) > 1:
                # Linear regression to estimate time complexity
                sizes = np.array(data['sizes'])
                times = np.array(data['times'])
                
                # Try different complexity models
                linear_fit = np.polyfit(sizes, times, 1)
                quadratic_fit = np.polyfit(sizes, times, 2)
                
                linear_r2 = np.corrcoef(sizes, times)[0, 1] ** 2
                
                print(f"  Scalability analysis:")
                print(f"    Linear fit R²: {linear_r2:.3f}")
                print(f"    Growth rate: {linear_fit[0]:.6f} s/message")
        
        # Save results to file
        with open('performance_benchmark_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to 'performance_benchmark_results.json'")
    
    def create_performance_plots(self):
        """Create performance visualization plots"""
        try:
            print("📈 Creating performance plots...")
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('COMPOTE Performance Benchmarks', fontsize=16)
            
            # Plot 1: Execution Time vs Input Size
            ax1 = axes[0, 0]
            for algorithm, data in self.results.items():
                if data['sizes'] and data['times']:
                    ax1.plot(data['sizes'], data['times'], marker='o', label=algorithm)
            ax1.set_xlabel('Input Size (messages)')
            ax1.set_ylabel('Execution Time (seconds)')
            ax1.set_title('Execution Time vs Input Size')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Throughput vs Input Size
            ax2 = axes[0, 1]
            for algorithm, data in self.results.items():
                if data['sizes'] and data['throughput']:
                    ax2.plot(data['sizes'], data['throughput'], marker='s', label=algorithm)
            ax2.set_xlabel('Input Size (messages)')
            ax2.set_ylabel('Throughput (operations/second)')
            ax2.set_title('Throughput vs Input Size')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Algorithm Comparison (Average Times)
            ax3 = axes[1, 0]
            algorithms = []
            avg_times = []
            for algorithm, data in self.results.items():
                if data['times']:
                    algorithms.append(algorithm)
                    avg_times.append(statistics.mean(data['times']))
            
            if algorithms:
                bars = ax3.bar(algorithms, avg_times)
                ax3.set_ylabel('Average Execution Time (seconds)')
                ax3.set_title('Algorithm Performance Comparison')
                ax3.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, time in zip(bars, avg_times):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height,
                            f'{time:.3f}s', ha='center', va='bottom')
            
            # Plot 4: Scalability Analysis
            ax4 = axes[1, 1]
            for algorithm, data in self.results.items():
                if len(data['sizes']) > 2 and data['times']:
                    sizes = np.array(data['sizes'])
                    times = np.array(data['times'])
                    
                    # Normalize to show relative scaling
                    normalized_times = times / times[0]
                    normalized_sizes = sizes / sizes[0]
                    
                    ax4.plot(normalized_sizes, normalized_times, marker='d', label=algorithm)
            
            ax4.set_xlabel('Relative Input Size')
            ax4.set_ylabel('Relative Execution Time')
            ax4.set_title('Scalability Analysis (Normalized)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('compote_performance_benchmarks.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✅ Performance plots saved to 'compote_performance_benchmarks.png'")
            
        except ImportError:
            print("⚠️ matplotlib not available, skipping plot generation")
        except Exception as e:
            print(f"❌ Error creating plots: {e}")


def main():
    """Main benchmarking function"""
    print("🏁 COMPOTE Performance Benchmarking Suite")
    print("Copyright (c) 2024 COMPOTE Implementation")
    print()
    
    benchmark = CompotePerformanceBenchmark()
    
    try:
        benchmark.run_all_benchmarks()
        print("\n🎉 All benchmarks completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Benchmarking interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmarking failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()