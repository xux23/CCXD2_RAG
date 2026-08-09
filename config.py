"""
RAG系统配置文件
"""

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Any

# 项目根目录（config.py 位于项目根目录）
_PROJECT_ROOT = Path(__file__).resolve().parent

@dataclass
class RAGConfig:
    """RAG系统配置类"""

    # 路径配置（基于 __file__ 解析为绝对路径，避免依赖运行时工作目录）
    data_path: str = str(_PROJECT_ROOT / "data" / "C8" / "cook")
    index_save_path: str = str(_PROJECT_ROOT / "vector_index")

    # 模型配置
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
    llm_model: str = os.getenv("LLM_MODEL", "qwen-plus")

    # 检索配置
    top_k: int = 3

    # 生成配置
    temperature: float = 0.1
    max_tokens: int = 2048

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RAGConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

# 默认配置实例
DEFAULT_CONFIG = RAGConfig()
