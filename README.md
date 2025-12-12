# COMPOTE - COntext-aware Message seed PriOritization and muTation in consEnsus fuzzing

A comprehensive implementation of the COMPOTE fuzzing engine for consensus protocol testing, based on the research paper algorithms with full modular architecture and plugin support.

## 🎯 Overview

COMPOTE is an advanced fuzzing engine specifically designed for consensus protocols that implements:

- **Algorithm 1**: Selective & Conditional Parsing (O(1) per message)
- **Algorithm 2**: Feature Extraction (O(M·f) complexity)  
- **Algorithm 3**: Context Clustering using DBSCAN (O(n·log n) average)
- **Algorithm 4**: Priority Calculation with weighted scoring (O(n·|T|) with pruning)
- **Algorithm 5**: State Analysis & Priority Update with feedback loops

## 🚀 Key Features

### Core Algorithms
- ✅ **Selective Parsing**: Efficiently extracts only critical fields from consensus messages
- ✅ **Feature Extraction**: Converts messages into numerical vectors for clustering
- ✅ **Context Clustering**: Groups similar messages using DBSCAN with Euclidean distance
- ✅ **Priority Calculation**: Weighted scoring combining similarity, fault history, and coverage
- ✅ **State Analysis**: Drives fuzzing iterations with continuous refinement

### Architecture
- 🔧 **Modular Design**: Each algorithm implemented as discrete, reusable component
- 🔌 **Plugin System**: Seamless integration with LOKI and Tyr frameworks
- ⚡ **Performance Optimized**: O(n·log n) overall complexity with caching and pruning
- 🎛️ **Configurable**: Extensive configuration options for all algorithms
- 📊 **Monitoring**: Real-time statistics and performance metrics

### Advanced Features
- 🤖 **Simulation Mode**: Test without real consensus networks
- 🔄 **Auto-optimization**: Dynamic parameter tuning based on performance
- 📈 **Coverage Tracking**: Code coverage and new path discovery
- 🐛 **Fault Detection**: Comprehensive fault categorization and analysis
- 💾 **State Persistence**: Save/load fuzzing campaigns
- 🎨 **Visualization**: Cluster analysis and performance plots

## 📦 Installation

### Requirements
```bash
# Python 3.7+
pip install -r requirements.txt
```

### Dependencies
- `numpy>=1.21.0` - Numerical computations
- `scikit-learn>=1.0.0` - Machine learning algorithms (DBSCAN)
- `pandas>=1.3.0` - Data manipulation
- `matplotlib>=3.5.0` - Visualization (optional)
- `pytest>=6.2.0` - Testing framework

### Quick Install
```bash
git clone <repository>
cd compote
pip install -r requirements.txt
```

## 🔧 Quick Start

### Basic Usage
```python
from compote import CompoteFuzzer
from compote.core.types import RawMessage
import json
import time

# Create fuzzer with configuration
config = {
    'simulation_mode': True,
    'max_iterations': 100,
    'clustering_eps': 0.3,
    'priority_alpha': 0.3,  # similarity weight
    'priority_beta': 0.4,   # fault weight  
    'priority_gamma': 0.3   # coverage weight
}

with CompoteFuzzer(config) as fuzzer:
    # Load consensus messages
    messages = [
        {
            'message_type': 'propose',
            'round_number': 1,
            'sender_id': 'leader_node',
            'block_hash': 'block_123'
        }
        # ... more messages
    ]
    
    fuzzer.load_messages(messages)
    fuzzer.initialize_seed_pool()
    
    # Start fuzzing
    fuzzer.start_fuzzing()
    
    # Get results
    report = fuzzer.get_comprehensive_report()
    print(f"Faults found: {report['summary']['faults_discovered']}")
```

### Plugin Integration
```python
from compote.plugins import LokiPlugin, TyrPlugin
from compote.fabric import FabricPlugin

# LOKI integration
loki_plugin = LokiPlugin({
    'loki_host': 'localhost',
    'loki_port': 9001,
    'seed_batch_size': 10
})

# Tyr integration  
tyr_plugin = TyrPlugin({
    'tyr_path': '/usr/local/bin/tyr',
    'working_dir': '/tmp/compote_tyr'
})

# Hyperledger Fabric v2.5 integration
fabric_plugin = FabricPlugin({
    'fabric_version': '2.5',
    'network_name': 'compote-fabric',
    'rest_api_base_url': 'http://localhost:4000',
    'orderer_endpoints': ['localhost:7050'],
    'peer_endpoints': ['localhost:7051', 'localhost:8051']
})
```

## 📖 Examples

### Run Basic Example
```bash
python examples/basic_usage.py
```

### Advanced Features Demo
```bash
python examples/advanced_usage.py
```

### Performance Benchmarking
```bash
python examples/performance_benchmark.py
```

## 🧪 Testing

### Run All Tests
```bash
python tests/run_tests.py
```

### Coverage Analysis
```bash
python tests/run_tests.py --coverage
```

### Performance Tests
```bash
python tests/run_tests.py --performance
```

### Specific Test Patterns
```bash
python tests/run_tests.py --pattern "test_parser*"
```



## ⚙️ Configuration

### Core Configuration
```python
config = {
    # Feature extraction
    'normalize_features': True,
    
    # Clustering  
    'clustering_eps': 0.5,
    'clustering_min_samples': 3,
    'scale_features': True,
    
    # Priority calculation
    'priority_alpha': 0.3,  # similarity weight
    'priority_beta': 0.4,   # fault weight
    'priority_gamma': 0.3,  # coverage weight
    'fault_threshold': 0.1,
    'similarity_threshold': 0.2,
    'time_threshold': 3600.0,
    
    # Mutation
    'timestamp_variance': 0.1,
    'round_variance': 2,
    'view_variance': 1,
    'enabled_mutations': ['timestamp', 'round', 'view', 'payload'],
    
    # Execution
    'simulation_mode': False,
    'execution_timeout': 30.0,
    'max_retries': 3,
    
    # Performance
    'max_workers': 4,
    'max_iterations': 1000,
    'save_interval': 100,
    'auto_optimize': True
}
```
