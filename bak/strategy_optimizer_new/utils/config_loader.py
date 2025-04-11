import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"加载配置文件时出错: {str(e)}")
        return {} 