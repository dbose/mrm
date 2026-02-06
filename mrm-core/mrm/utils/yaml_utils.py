"""Utility functions for loading and validating YAML configuration files"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import os


def load_yaml(file_path: Path) -> Dict[str, Any]:
    """Load YAML file with environment variable substitution"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Simple environment variable substitution
    # Supports {{ env_var('VAR_NAME') }} syntax
    import re
    pattern = r'\{\{\s*env_var\([\'"]([^\'"]+)[\'"]\)\s*\}\}'
    
    def replace_env(match):
        var_name = match.group(1)
        return os.environ.get(var_name, '')
    
    content = re.sub(pattern, replace_env, content)
    
    return yaml.safe_load(content)


def save_yaml(data: Dict[str, Any], file_path: Path) -> None:
    """Save dictionary to YAML file"""
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def find_project_root(start_path: Optional[Path] = None) -> Optional[Path]:
    """
    Find project root by looking for mrm_project.yml
    
    Args:
        start_path: Path to start searching from (defaults to cwd)
    
    Returns:
        Path to project root or None if not found
    """
    if start_path is None:
        start_path = Path.cwd()
    
    current = start_path.resolve()
    
    # Walk up directory tree
    while current != current.parent:
        if (current / 'mrm_project.yml').exists():
            return current
        current = current.parent
    
    return None


def validate_model_config(config: Dict[str, Any]) -> None:
    """Validate model configuration structure"""
    required_fields = ['model']
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in model config: {field}")
    
    model_config = config['model']
    required_model_fields = ['name', 'version']
    
    for field in required_model_fields:
        if field not in model_config:
            raise ValueError(f"Missing required field in model: {field}")


def validate_project_config(config: Dict[str, Any]) -> None:
    """Validate project configuration structure"""
    required_fields = ['name', 'version']
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in project config: {field}")
