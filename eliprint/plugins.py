# eliprint/plugins.py
"""
Plugin system for extending EliPrint functionality.
"""

import importlib
import inspect
import logging
from typing import Dict, Any, Optional, Callable, List, Type

logger = logging.getLogger(__name__)

class PluginBase:
    """Base class for all plugins."""
    
    def __init__(self, **kwargs):
        """Initialize the plugin with optional parameters."""
        self.name = self.__class__.__name__
        self.description = self.__doc__ or "No description available"
        self.params = kwargs
    
    def get_info(self) -> Dict[str, Any]:
        """Get plugin information."""
        return {
            'name': self.name,
            'description': self.description,
            'params': self.params
        }


class FingerprintPlugin(PluginBase):
    """Base class for fingerprint plugins."""
    
    def extract(self, audio_data, **kwargs) -> List[Dict[str, Any]]:
        """
        Extract fingerprints from audio data.
        
        Args:
            audio_data: Audio data as NumPy array
            **kwargs: Additional parameters
            
        Returns:
            List of fingerprint data
        """
        raise NotImplementedError("Plugin must implement extract method")


class PluginManager:
    """Manager for loading and using plugins."""
    
    def __init__(self):
        """Initialize the plugin manager."""
        self.plugins = {}
        self.fingerprint_plugins = {}
    
    def register_plugin(self, plugin_class: Type[PluginBase]):
        """
        Register a plugin class.
        
        Args:
            plugin_class: Plugin class to register
        """
        # Create instance
        plugin = plugin_class()
        
        # Register based on type
        if issubclass(plugin_class, FingerprintPlugin):
            self.fingerprint_plugins[plugin.name] = plugin_class
        
        # Register in general plugins
        self.plugins[plugin.name] = plugin_class
        
        logger.info(f"Registered plugin: {plugin.name}")
    
    def load_plugin_from_module(self, module_name: str):
        """
        Load plugins from a module.
        
        Args:
            module_name: Name of the module to load
        """
        try:
            # Import module
            module = importlib.import_module(module_name)
            
            # Find plugin classes
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, PluginBase) and 
                    obj is not PluginBase and
                    obj is not FingerprintPlugin):
                    # Register plugin
                    self.register_plugin(obj)
            
        except Exception as e:
            logger.error(f"Error loading plugins from module '{module_name}': {str(e)}")
    
    def get_fingerprint_plugin(self, name: str) -> Optional[Type[FingerprintPlugin]]:
        """
        Get a fingerprint plugin by name.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin class or None if not found
        """
        return self.fingerprint_plugins.get(name)
    
    def list_plugins(self) -> Dict[str, Dict[str, Any]]:
        """
        List all registered plugins.
        
        Returns:
            Dictionary of plugin information
        """
        result = {}
        
        for name, plugin_class in self.plugins.items():
            plugin = plugin_class()
            result[name] = plugin.get_info()
        
        return result

