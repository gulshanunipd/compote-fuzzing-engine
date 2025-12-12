"""
COMPOTE Plugins Module

Contains plugin interfaces for integrating with other fuzzing frameworks.
"""

from .base_plugin import PluginInterface, FrameworkAdapter
from .loki_plugin import LokiPlugin
from .tyr_plugin import TyrPlugin

__all__ = [
    'PluginInterface',
    'FrameworkAdapter', 
    'LokiPlugin',
    'TyrPlugin'
]