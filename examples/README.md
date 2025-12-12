# COMPOTE Examples

This directory contains example scripts demonstrating various aspects of the COMPOTE fuzzing engine.

## Examples Overview

### 1. Basic Usage (`basic_usage.py`)
A comprehensive introduction to COMPOTE showing:
- Loading consensus messages
- Initializing the fuzzing engine
- Running a basic fuzzing campaign
- Analyzing results and statistics
- Demonstrating clustering capabilities

**Run with:**
```bash
python examples/basic_usage.py
```

### 2. Advanced Usage (`advanced_usage.py`)
Advanced features demonstration including:
- Plugin integration with LOKI/Tyr frameworks
- Custom mutation strategies
- Real-time monitoring and analysis
- Complex message scenario generation
- Performance optimization techniques

**Run with:**
```bash
python examples/advanced_usage.py
```

### 3. Performance Benchmarking (`performance_benchmark.py`)
Comprehensive performance analysis:
- Algorithm-specific benchmarking
- Scalability analysis
- Throughput measurements
- Performance visualization
- Comparative analysis across different input sizes

**Run with:**
```bash
python examples/performance_benchmark.py
```

## Prerequisites

Before running the examples, ensure you have:

1. **Required Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Optional Dependencies (for advanced features):**
   ```bash
   pip install matplotlib seaborn  # For visualization
   ```

3. **Framework Integration (optional):**
   - LOKI framework (for `advanced_usage.py` with LOKI plugin)
   - Tyr framework (for `advanced_usage.py` with Tyr plugin)

## Example Outputs

### Basic Usage Output
```
🚀 COMPOTE Basic Fuzzing Example
==================================================
📋 Configuration:
  normalize_features: True
  clustering_eps: 0.3
  clustering_min_samples: 3
  ...
📨 Loaded 30 messages
✅ Initialized seed pool
📊 Initial Statistics:
  Messages processed: 30
  Clusters created: 4
  Active clusters: 4
🎯 Starting fuzzing campaign...
⚡ Progress: 20.0% (Iteration 10/50)
✅ Fuzzing completed in 12.34 seconds
```

### Advanced Usage Features
- Real-time monitoring dashboards
- Plugin integration status
- Complex scenario testing
- Fault analysis and reporting

### Performance Benchmark Results
- Algorithm performance comparison
- Scalability analysis charts
- Throughput measurements
- JSON export of detailed metrics

## Configuration Options

### Basic Configuration
```python
config = {
    'normalize_features': True,
    'clustering_eps': 0.3,
    'clustering_min_samples': 3,
    'priority_alpha': 0.3,  # similarity weight
    'priority_beta': 0.4,   # fault weight
    'priority_gamma': 0.3,  # coverage weight
    'simulation_mode': True,
    'max_iterations': 100
}
```

### Advanced Configuration
```python
advanced_config = {
    # Enhanced clustering
    'scale_features': True,
    'auto_optimize': True,
    
    # Dynamic thresholds
    'fault_threshold': 0.05,
    'similarity_threshold': 0.15,
    'time_threshold': 1800.0,
    
    # Performance tuning
    'max_workers': 4,
    'save_interval': 50,
    
    # Plugin settings
    'loki_host': 'localhost',
    'loki_port': 9001,
    'tyr_path': '/usr/local/bin/tyr'
}
```

## Customization

### Adding Custom Message Types
```python
def generate_custom_messages():
    custom_data = {
        'message_type': 'custom_consensus',
        'custom_field': 'custom_value',
        # ... other fields
    }
    return [RawMessage(json.dumps(custom_data).encode(), time.time(), 'node_1')]
```

### Custom Mutation Strategies
```python
mutation_strategy = MutationStrategy(
    timestamp_variance=0.2,
    round_variance=5,
    enabled_mutations=['timestamp', 'round', 'custom_field']
)
```

### Custom Callbacks
```python
def custom_fault_callback(result):
    print(f"Custom fault handler: {result.message_id}")
    # Custom fault analysis logic
    
fuzzer.set_fault_callback(custom_fault_callback)
```

## Troubleshooting

### Common Issues

1. **Import Errors:**
   ```bash
   # Ensure COMPOTE is in Python path
   export PYTHONPATH="${PYTHONPATH}:/path/to/compote"
   ```

2. **Plugin Connection Failures:**
   - Verify LOKI/Tyr frameworks are running
   - Check network connectivity and ports
   - Review plugin configuration

3. **Performance Issues:**
   - Reduce message set size for testing
   - Adjust worker count based on system resources
   - Enable simulation mode for development

4. **Memory Usage:**
   - Limit max_iterations for large datasets
   - Use save_interval to checkpoint progress
   - Monitor historical record limits

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Integration Examples

### With Existing Test Suites
```python
# Integration with pytest
def test_consensus_with_compote():
    fuzzer = CompoteFuzzer({'simulation_mode': True})
    # ... fuzzing logic
    assert fuzzer.get_comprehensive_report()['summary']['faults_discovered'] < threshold
```

### With CI/CD Pipelines
```bash
# Add to CI script
python examples/basic_usage.py --output-format=json > fuzzing_results.json
if [ $(jq '.summary.faults_discovered' fuzzing_results.json) -gt 0 ]; then
    echo "Faults detected, failing build"
    exit 1
fi
```

## Further Reading

- [COMPOTE Research Paper](../COMP.pdf) - Original research and algorithms
- [API Documentation](../docs/api.md) - Detailed API reference
- [Architecture Guide](../docs/architecture.md) - System design overview
- [Plugin Development](../docs/plugins.md) - Creating custom plugins

## Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the main documentation
3. Create an issue with detailed error information