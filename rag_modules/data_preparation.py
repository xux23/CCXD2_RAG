"""
数据准备模块 — 文档加载、元数据增强、Markdown 分块
"""

import hashlib
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

logger = logging.getLogger(__name__)


class DataPreparationModule:
    """数据准备模块：负责食谱文档的加载、元数据增强和 Markdown 分块"""

    # ------------------------------------------------------------------
    # 常量：分类 & 难度映射
    # ------------------------------------------------------------------

    CATEGORY_MAPPING: Dict[str, str] = {
        "meat_dish": "荤菜",
        "vegetable_dish": "素菜",
        "soup": "汤品",
        "dessert": "甜品",
        "breakfast": "早餐",
        "staple": "主食",
        "aquatic": "水产",
        "condiment": "调料",
        "drink": "饮品",
        "semi-finished": "半成品",
    }

    # 需要跳过的目录（模板文件不应进入知识库）
    SKIP_DIRS: set = {"template"}

    DIFFICULTY_MAP: Dict[int, str] = {
        5: "非常困难",
        4: "困难",
        3: "中等",
        2: "简单",
        1: "非常简单",
    }

    # 类方法：对外暴露支持的分类/难度列表
    @classmethod
    def get_supported_categories(cls) -> List[str]:
        """返回所有支持的菜品分类（中文）"""
        return list(cls.CATEGORY_MAPPING.values())

    @classmethod
    def get_supported_difficulties(cls) -> List[str]:
        """返回所有支持的难度等级"""
        return list(cls.DIFFICULTY_MAP.values())

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.documents: List[Document] = []
        self.chunks: List[Document] = []
        # 懒加载索引：首次调用 get_parent_documents 时构建
        self._parent_docs_index: Optional[Dict[str, Document]] = None

    # ------------------------------------------------------------------
    # 文档加载
    # ------------------------------------------------------------------

    def load_documents(self) -> List[Document]:
        """递归扫描 data_path 下所有 .md 文件，构建父文档列表"""
        documents: List[Document] = []
        data_path_obj = Path(self.data_path)

        for md_file in data_path_obj.rglob("*.md"):
            # 跳过模板目录（如 template/示例菜）
            if any(part in self.SKIP_DIRS for part in md_file.parts):
                continue

            # 规范化路径：无论 data_path 以相对/绝对方式传入，ID 与 source 保持一致
            md_file = md_file.resolve()
            content = md_file.read_text(encoding="utf-8")
            doc = Document(
                page_content=content,
                metadata={
                    "source": str(md_file),
                    # 确定性 ID：基于文件路径哈希，保证重载持久化索引后 ID 一致
                    "parent_id": hashlib.md5(str(md_file).encode("utf-8")).hexdigest(),
                    "doc_type": "parent",
                },
            )
            documents.append(doc)

        for doc in documents:
            self._enhance_metadata(doc)

        # 重置懒加载索引（文档列表已更新）
        self._parent_docs_index = None
        self.documents = documents
        logger.info("已加载 %d 篇文档", len(documents))
        return documents

    # ------------------------------------------------------------------
    # 元数据增强
    # ------------------------------------------------------------------

    def _enhance_metadata(self, doc: Document):
        """根据文件路径和内容提取分类、菜名、难度等元数据"""
        file_path = Path(doc.metadata.get("source", ""))

        # 菜品分类：从路径中匹配
        doc.metadata["category"] = "其他"
        for part in file_path.parts:
            if part in self.CATEGORY_MAPPING:
                doc.metadata["category"] = self.CATEGORY_MAPPING[part]
                break

        # 菜名：取文件名（不含扩展名）
        doc.metadata["dish_name"] = file_path.stem

        # 难度：从内容中的 ★ 符号推断
        star_match = re.search(r"★+", doc.page_content)
        if star_match:
            star_count = len(star_match.group())
            doc.metadata["difficulty"] = self.DIFFICULTY_MAP.get(star_count, "未知")
        else:
            doc.metadata["difficulty"] = "未知"

    # ------------------------------------------------------------------
    # 文档分块
    # ------------------------------------------------------------------

    def chunk_documents(self) -> List[Document]:
        """使用 Markdown 标题分割器将文档分块，自动过滤空块"""
        if not self.documents:
            raise ValueError("请先调用 load_documents() 加载文档")

        raw_chunks = self._markdown_header_split()

        # 过滤空文本块（MarkdownHeaderTextSplitter 可能产生空 chunk）
        chunks = [
            c for c in raw_chunks
            if c.page_content and c.page_content.strip()
        ]
        skipped = len(raw_chunks) - len(chunks)
        if skipped:
            logger.warning("过滤掉 %d 个空文本块（原始 %d 个）", skipped, len(raw_chunks))

        self.chunks = chunks
        logger.info("共生成 %d 个有效文本块", len(chunks))
        return chunks

    def _markdown_header_split(self) -> List[Document]:
        """按 Markdown 标题层级（#/##/###）切分文档"""
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "主标题"),
                ("##", "二级标题"),
                ("###", "三级标题"),
            ],
            strip_headers=False,
        )

        all_chunks: List[Document] = []
        for doc in self.documents:
            md_chunks = splitter.split_text(doc.page_content)
            parent_id = doc.metadata["parent_id"]

            for i, chunk in enumerate(md_chunks):
                # 确定性 chunk_id：基于 parent_id + 块序号，重载索引后与持久化块保持一致
                child_id = hashlib.md5(f"{parent_id}:{i}".encode("utf-8")).hexdigest()
                chunk.metadata.update(doc.metadata)
                chunk.metadata.update(
                    {
                        "chunk_id": child_id,
                        "batch_index": i,
                        "chunk_size": len(chunk.page_content),
                        "parent_id": parent_id,
                    }
                )

            all_chunks.extend(md_chunks)

        return all_chunks

    # ------------------------------------------------------------------
    # 父文档查找（懒索引，O(1) 查找）
    # ------------------------------------------------------------------

    def _ensure_parent_index(self):
        """构建 parent_id → Document 的索引（仅首次调用时构建）"""
        if self._parent_docs_index is None:
            self._parent_docs_index = {
                doc.metadata["parent_id"]: doc for doc in self.documents
            }

    def get_parent_documents(
        self, child_chunks: List[Document]
    ) -> List[Document]:
        """根据子块反查父文档，按相关度（命中次数）降序返回，自动去重"""
        self._ensure_parent_index()

        parent_hit_count: Dict[str, int] = {}
        for chunk in child_chunks:
            pid = chunk.metadata.get("parent_id")
            if pid:
                parent_hit_count[pid] = parent_hit_count.get(pid, 0) + 1

        sorted_ids = sorted(parent_hit_count, key=parent_hit_count.get, reverse=True)
        return [
            self._parent_docs_index[pid]
            for pid in sorted_ids
            if pid in self._parent_docs_index
        ]

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """返回知识库统计信息"""
        categories: Dict[str, int] = {}
        difficulties: Dict[str, int] = {}
        for doc in self.documents:
            cat = doc.metadata.get("category", "其他")
            categories[cat] = categories.get(cat, 0) + 1
            diff = doc.metadata.get("difficulty", "未知")
            difficulties[diff] = difficulties.get(diff, 0) + 1

        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "categories": categories,
            "difficulties": difficulties,
        }
