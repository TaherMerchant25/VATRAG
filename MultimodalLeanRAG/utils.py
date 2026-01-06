"""
Utility Functions for Multimodal LeanRAG
========================================
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def write_jsonl(data: List[Dict], filepath: str, mode: str = 'w'):
    """Write data to JSONL file."""
    with open(filepath, mode, encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def read_jsonl(filepath: str) -> List[Dict]:
    """Read JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def ensure_dir(path: str):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)


def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_time(seconds: float) -> str:
    """Format seconds to MM:SS.ms format."""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"


def batch_iterator(items: List, batch_size: int):
    """Yield batches from a list."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


class ConfigManager:
    """Configuration manager with environment variable support."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = {}
        if config_path and os.path.exists(config_path):
            self.load(config_path)
    
    def load(self, config_path: str):
        """Load configuration from YAML file."""
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with dot notation support."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default
    
    def set(self, key: str, value: Any):
        """Set config value with dot notation support."""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        config[keys[-1]] = value


class ProgressTracker:
    """Track processing progress."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.now()
    
    def update(self, n: int = 1):
        """Update progress."""
        self.current += n
    
    def get_progress(self) -> Dict[str, Any]:
        """Get progress information."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.current / elapsed if elapsed > 0 else 0
        eta = (self.total - self.current) / rate if rate > 0 else 0
        
        return {
            "current": self.current,
            "total": self.total,
            "percentage": (self.current / self.total * 100) if self.total > 0 else 0,
            "elapsed_seconds": elapsed,
            "rate_per_second": rate,
            "eta_seconds": eta
        }
