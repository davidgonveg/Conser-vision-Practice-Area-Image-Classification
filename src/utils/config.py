"""
Taï National Park Species Classification - Configuration Management

This module provides configuration management utilities for the project,
including loading YAML configs, merging with command line arguments,
and providing default configurations.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class Config:
    """
    Configuration management class for the Taï Park project.
    
    Handles loading, merging, and accessing configuration parameters
    from YAML files with support for nested keys and environment variables.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config = self._load_default_config()
        
        if config_path:
            self.load_config_file(config_path)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration values."""
        return {
            'data': {
                'raw_dir': 'data/raw',
                'processed_dir': 'data/processed',
                'train_images_dir': 'data/raw/train_features',
                'test_images_dir': 'data/raw/test_features',
                'train_metadata': 'data/raw/train_features.csv',
                'test_metadata': 'data/raw/test_features.csv',
                'train_labels': 'data/raw/train_labels.csv'
            },
            'model': {
                'name': 'resnet50',
                'num_classes': 8,
                'pretrained': True,
                'dropout': 0.5
            },
            'training': {
                'batch_size': 32,
                'num_epochs': 50,
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'optimizer': 'adam',
                'momentum': 0.9,
                'validation_split': 0.2,
                'early_stopping_patience': 10,
                'gradient_clip': 1.0,
                'sampler_type': 'weighted',
                'num_workers': 4
            },
            'scheduler': {
                'type': 'plateau',
                'patience': 5,
                'factor': 0.5,
                'step_size': 10,
                'gamma': 0.1
            },
            'image': {
                'size': [224, 224],
                'normalize': {
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]
                }
            },
            'augmentation': {
                'horizontal_flip': 0.5,
                'vertical_flip': 0.2,
                'rotation': 15,
                'brightness': 0.2,
                'contrast': 0.2,
                'color_jitter': 0.2,
                'aggressive': False
            },
            'loss': {
                'type': 'cross_entropy',
                'focal_alpha': 1.0,
                'focal_gamma': 2.0,
                'label_smoothing': 0.1,
                'use_class_weights': False
            },
            'hardware': {
                'device': 'auto',
                'mixed_precision': False,
                'compile': False,
                'deterministic': False
            },
            'logging': {
                'log_dir': 'results/logs',
                'tensorboard_dir': 'results/logs/tensorboard',
                'log_level': 'INFO',
                'save_frequency': 5,
                'log_frequency': 100
            },
            'output': {
                'models_dir': 'results/models',
                'predictions_dir': 'data/submissions',
                'plots_dir': 'results/plots'
            },
            'classes': [
                'antelope_duiker', 'bird', 'blank', 'civet_genet',
                'hog', 'leopard', 'monkey_prosimian', 'rodent'
            ],
            'random_state': 42
        }
    
    def load_config_file(self, config_path: Union[str, Path]) -> None:
        """
        Load configuration from YAML file and merge with existing config.
        
        Args:
            config_path: Path to YAML configuration file
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return
        
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
            
            if file_config:
                self.config = self._merge_configs(self.config, file_config)
                logger.info(f"Loaded configuration from: {config_path}")
            else:
                logger.warning(f"Empty config file: {config_path}")
                
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            raise
    
    def _merge_configs(self, base_config: Dict[str, Any], new_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge two configuration dictionaries.
        
        Args:
            base_config: Base configuration dictionary
            new_config: New configuration to merge
            
        Returns:
            Merged configuration dictionary
        """
        merged = base_config.copy()
        
        for key, value in new_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key using dot notation (e.g., 'model.name')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key using dot notation
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent of the final key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the final value
        config[keys[-1]] = value
    
    def update_from_args(self, args) -> None:
        """
        Update configuration with command line arguments.
        
        Args:
            args: Parsed command line arguments
        """
        # Map command line arguments to config keys
        arg_mappings = {
            # Model settings
            'model': 'model.name',
            'pretrained': 'model.pretrained',
            'dropout': 'model.dropout',
            
            # Training settings
            'epochs': 'training.num_epochs',
            'batch_size': 'training.batch_size',
            'learning_rate': 'training.learning_rate',
            'weight_decay': 'training.weight_decay',
            'optimizer': 'training.optimizer',
            'momentum': 'training.momentum',
            'early_stopping': 'training.early_stopping_patience',
            'gradient_clip': 'training.gradient_clip',
            'num_workers': 'training.num_workers',
            'sampler': 'training.sampler_type',
            
            # Data settings
            'data_dir': 'data.raw_dir',
            'image_size': 'image.size',
            'fraction': 'data.fraction',
            'random_state': 'random_state',
            
            # Scheduler settings
            'scheduler': 'scheduler.type',
            'scheduler_patience': 'scheduler.patience',
            'scheduler_factor': 'scheduler.factor',
            'scheduler_step_size': 'scheduler.step_size',
            'scheduler_gamma': 'scheduler.gamma',
            
            # Loss settings
            'loss': 'loss.type',
            'focal_alpha': 'loss.focal_alpha',
            'focal_gamma': 'loss.focal_gamma',
            'label_smoothing': 'loss.label_smoothing',
            'class_weights': 'loss.use_class_weights',
            
            # Augmentation settings
            'aggressive_aug': 'augmentation.aggressive',
            'horizontal_flip': 'augmentation.horizontal_flip',
            'rotation': 'augmentation.rotation',
            'brightness': 'augmentation.brightness',
            'contrast': 'augmentation.contrast',
            'color_jitter': 'augmentation.color_jitter',
            
            # Hardware settings
            'device': 'hardware.device',
            'mixed_precision': 'hardware.mixed_precision',
            'compile': 'hardware.compile',
            'deterministic': 'hardware.deterministic',
            
            # Logging settings
            'log_level': 'logging.log_level',
            'save_frequency': 'logging.save_frequency',
            'log_frequency': 'logging.log_frequency'
        }
        
        # Update config with argument values
        for arg_name, config_key in arg_mappings.items():
            if hasattr(args, arg_name):
                arg_value = getattr(args, arg_name)
                if arg_value is not None:
                    # Special handling for image_size (convert to [size, size])
                    if arg_name == 'image_size' and isinstance(arg_value, int):
                        arg_value = [arg_value, arg_value]
                    
                    self.set(config_key, arg_value)
    
    def save(self, output_path: Union[str, Path]) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            output_path: Path to save configuration
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to: {output_path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get configuration as dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy()
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return yaml.dump(self.config, default_flow_style=False, indent=2)
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config({len(self.config)} sections)"


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Factory function to load configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config instance
    """
    return Config(config_path)


def create_config_from_args(args) -> Config:
    """
    Create configuration from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Config instance with arguments applied
    """
    # Load base config file if specified
    config = Config(getattr(args, 'config', None))
    
    # Update with command line arguments
    config.update_from_args(args)
    
    return config


# Environment variable substitution
def substitute_env_vars(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Substitute environment variables in configuration values.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Configuration with environment variables substituted
    """
    def _substitute_value(value):
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            env_var = value[2:-1]
            default_value = None
            
            if ':' in env_var:
                env_var, default_value = env_var.split(':', 1)
            
            return os.getenv(env_var, default_value)
        elif isinstance(value, dict):
            return {k: _substitute_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_substitute_value(item) for item in value]
        else:
            return value
    
    return {k: _substitute_value(v) for k, v in config_dict.items()}


# Example usage and testing
if __name__ == "__main__":
    # Test configuration loading
    print("Testing Config class...")
    
    # Create default config
    config = Config()
    print(f"Default model: {config.get('model.name')}")
    print(f"Default batch size: {config.get('training.batch_size')}")
    
    # Test setting values
    config.set('model.name', 'efficientnet_b3')
    config.set('training.batch_size', 64)
    
    print(f"Updated model: {config.get('model.name')}")
    print(f"Updated batch size: {config.get('training.batch_size')}")
    
    # Test with missing key
    print(f"Missing key: {config.get('nonexistent.key', 'default_value')}")
    
    print("Config class working correctly!")