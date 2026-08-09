"""
检索优化模块 — 向量检索 + BM25 混合检索 + RRF 重排
"""

import logging
from typing import List, Dict, Any

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class RetrievalOptimizationModule:
    """检索优化模块：混合检索（向量 + BM25）+ RRF 重排 + 元数据过滤"""

    def __init__(self, vectorstore: FAISS, chunks: List[Document]):
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.vector_retriever = None   # 向量检索器
        self.bm25_retriever = None     # BM25 检索器
        self._setup_retrievers()

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def _setup_retrievers(self):
        """初始化向量检索器和 BM25 检索器"""
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )
        self.bm25_retriever = BM25Retriever.from_documents(self.chunks, k=5)

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def hybrid_search(self, query: str, top_k: int = 3) -> List[Document]:
        """混合检索：向量 + BM25，RRF 融合重排后取 top_k"""
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)
        reranked = self._rrf_rerank(vector_docs, bm25_docs)
        return reranked[:top_k]

    def metadata_filtered_search(
        self, query: str, filters: Dict[str, Any], top_k: int = 5
    ) -> List[Document]:
        """带元数据过滤的向量检索"""
        filtered_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k * 3, "filter": filters},
        )
        results = filtered_retriever.invoke(query)
        return results[:top_k]

    # ------------------------------------------------------------------
    # 内部：RRF 重排
    # ------------------------------------------------------------------

    @staticmethod
    def _rrf_rerank(
        vector_results: List[Document],
        bm25_results: List[Document],
        k: int = 60,
    ) -> List[Document]:
        """
        Reciprocal Rank Fusion (RRF) 融合两路检索结果。

        使用 chunk_id 作为文档唯一标识，确保同一文档在两路中出现时能正确合并分数。
        """
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for rank, doc in enumerate(vector_results):
            cid = doc.metadata.get("chunk_id", id(doc))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            doc_map[cid] = doc

        for rank, doc in enumerate(bm25_results):
            cid = doc.metadata.get("chunk_id", id(doc))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            doc_map.setdefault(cid, doc)

        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        return [doc_map[cid] for cid in sorted_ids]
