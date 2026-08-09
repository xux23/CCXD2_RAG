import logging
import os
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


class IndexConstructionModule:
    """索引构造类 负责向量化和索引构建"""
    def __init__(self, model_name: str = None,
                 index_save_path: str = "./vector_index"):
        # 默认从环境变量读取，回退到 text-embedding-v3
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
        self.index_save_path = index_save_path  # 索引保存路径
        self.embeddings = None  # 嵌入对象
        self.vectorstore = None  # 向量存储对象
        self.setup_embeddings()

    def setup_embeddings(self):
        """初始化嵌入模型（通过 OpenAI 兼容 API，复用 LLM_API_BASE / LLM_API_KEY）"""
        api_key = os.getenv("LLM_API_KEY") or os.getenv("MOONSHOT_API_KEY")
        api_base = os.getenv("LLM_API_BASE")
        if not api_key or not api_base:
            raise ValueError(
                "使用 API 嵌入模型需要设置 LLM_API_KEY 和 LLM_API_BASE 环境变量"
            )
        self.embeddings = OpenAIEmbeddings(
            model=self.model_name,
            api_key=api_key,
            base_url=api_base,
            # 阿里云百炼兼容接口只接受字符串数组，跳过 tiktoken 编码（默认会发送 token ID 数组导致 400）
            check_embedding_ctx_length=False,
            # 百炼 text-embedding-v3 单次请求批量上限 10 条
            chunk_size=10,
        )

    def build_vector_store(self, chunks: List[Document]):
        """构建向量索引"""
        if not chunks:
            raise ValueError("文档块列表不能为空")

        # 过滤掉空文本（MarkdownHeaderTextSplitter 可能产生空 chunk）
        valid_chunks = [c for c in chunks if c.page_content and c.page_content.strip()]
        skipped = len(chunks) - len(valid_chunks)
        if skipped:
            logger.warning("跳过 %d 个空文本块（共 %d 个）", skipped, len(chunks))

        if not valid_chunks:
            raise ValueError("所有文本块均为空，无法构建索引")

        texts = [chunk.page_content for chunk in valid_chunks]
        metadatas = [chunk.metadata for chunk in valid_chunks]

        self.vectorstore = FAISS.from_texts(
            texts=texts,
            metadatas=metadatas,
            embedding=self.embeddings,
        )
        logger.info("向量索引构建完成，共 %d 条记录", len(texts))
        return self.vectorstore

    def save_index(self):
        """保存向量索引到配置的路径"""
        if not self.vectorstore:
            raise ValueError("请先构建索引")

        #确保目录存在
        save_path = Path(self.index_save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(save_path))
        logger.info(f"索引已保存至: {save_path}")

    def load_index(self) -> Optional[FAISS]:
        """从配置路径加载向量索引，路径不存在或文件缺失时返回 None"""
        index_dir = Path(self.index_save_path)

        # 路径不存在，直接返回 None
        if not index_dir.exists():
            logger.info(f"索引目录不存在: {index_dir}，将重新构建")
            return None

        # 检查关键文件是否存在
        index_file = index_dir / "index.faiss"
        pkl_file = index_dir / "index.pkl"
        if not index_file.exists() or not pkl_file.exists():
            logger.info(f"索引文件不完整，将重新构建")
            return None

        try:
            self.vectorstore = FAISS.load_local(
                str(index_dir),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"成功加载索引: {index_dir}")
            return self.vectorstore
        except Exception as e:
            logger.warning(f"加载索引失败: {e}，将重新构建")
            return None