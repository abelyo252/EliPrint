
# eliprint/config.py
"""
Configuration management for EliPrint.
"""

from typing import Dict, Any, Optional
import json
import os
import logging

logger = logging.getLogger(__name__)

class Config:
    """
    Configuration manager for EliPrint.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Optional path to configuration file
        """
        # Default configuration
        self.fingerprinter_params = {
            'sample_rate': 44100,
            'window_size': 4096,
            'hop_size': 2048,
            'target_zone_size': 5,
            'target_zone_distance': 10,
            'min_freq': 100,
            'max_freq': 8000,
            'peak_neighborhood_size': 20,
            'peak_threshold': 0.3,
            'min_hash_time_delta': 0,
            'max_hash_time_delta': 200,
            'fingerprint_reduction': True
        }
        
        self.database_params = {
            'db_path': None,
            'in_memory': False,
            'batch_size': 1000,
            'auto_commit': True
        }
        
        self.recognizer_params = {
            'min_matches': 5,
            'min_confidence': 0.05,
            'time_tolerance': 0.5
        }
        
        self.batch_params = {
            'max_workers': None  # None means use all available CPU cores
        }
        
        # Load configuration from file if provided
        if config_path and os.path.exists(config_path):
            self.load(config_path)
    
    def update(self, **kwargs):
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        # Update specific parameter groups
        for param_group in ['fingerprinter_params', 'database_params', 'recognizer_params', 'batch_params']:
            if param_group in kwargs and isinstance(kwargs[param_group], dict):
                # Update parameter group
                getattr(self, param_group).update(kwargs[param_group])
    
    def save(self, config_path: str):
        """
        Save configuration to a file.
        
        Args:
            config_path: Path to save the configuration
        """
        config = {
            'fingerprinter_params': self.fingerprinter_params,
            'database_params': self.database_params,
            'recognizer_params': self.recognizer_params,
            'batch_params': self.batch_params
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        logger.info(f"Saved configuration to '{config_path}'")
    
    def load(self, config_path: str):
        """
        Load configuration from a file.
        
        Args:
            config_path: Path to the configuration file
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Update configuration
            self.update(**config)
            
            logger.info(f"Loaded configuration from '{config_path}'")
            
        except Exception as e:
            logger.error(f"Error loading configuration from '{config_path}': {str(e)}")
