"""
COMPOTE - COntext-aware Message seed PriOritization and muTation in consEnsus fuzzing

A modular fuzzing engine for consensus protocols that:
- Parses consensus messages to extract meaningful features
- Clusters messages into context pools using feature similarity  
- Assigns priorities based on similarity, historical faults, and code coverage
- Mutates high-priority seeds to generate test inputs
- Provides feedback-driven continuous refinement

Author: Implementation based on COMPOTE research paper
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "COMPOTE Implementation"

from .core.parser import MessageParser
from .core.feature_extractor import FeatureExtractor  
from .core.clustering import ContextClustering
from .core.priority_calculator import PriorityCalculator
from .core.state_analyzer import StateAnalyzer
from .engine.compote_fuzzer import CompoteFuzzer
from .plugins.loki_plugin import LokiPlugin
from .plugins.tyr_plugin import TyrPlugin
from .fabric.fabric_plugin import FabricPlugin

__all__ = [
    "MessageParser",
    "FeatureExtractor", 
    "ContextClustering",
    "PriorityCalculator",
    "StateAnalyzer",
    "CompoteFuzzer",
    "LokiPlugin",
    "TyrPlugin",
    "FabricPlugin"
]