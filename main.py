"""
尝尝咸淡 RAG 系统 — 主程序入口
"""

import logging
import os
import sys
from pathlib import Path

# 确保本目录在 sys.path 中，以便直接 python main.py 运行
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

# 必须在 import config 之前加载 .env（基于脚本所在目录，不依赖工作目录）
load_dotenv(Path(__file__).resolve().parent / ".env")

from config import DEFAULT_CONFIG, RAGConfig
from rag_modules import (
    DataPreparationModule,
    GenerationIntegrationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
)

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _doc_names(docs) -> list:
    """从文档列表中提取去重的菜品名称"""
    seen = set()
    names = []
    for doc in docs:
        name = doc.metadata.get("dish_name", "未知菜品")
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _format_chunk_info(chunk) -> str:
    """格式化单个 chunk 的简要信息（用于日志输出）"""
    dish = chunk.metadata.get("dish_name", "未知菜品")
    preview = chunk.page_content[:100].strip()
    if preview.startswith("#"):
        nl = preview.find("\n")
        title = preview[: nl if nl != -1 else len(preview)].replace("#", "").strip()
        return f"{dish}({title})"
    return f"{dish}(内容片段)"


# ---------------------------------------------------------------------------
# 主系统类
# ---------------------------------------------------------------------------

class RecipeRAGSystem:
    """食谱 RAG 系统主类"""

    def __init__(self, config: RAGConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.data_module: DataPreparationModule = None
        self.index_module: IndexConstructionModule = None
        self.retrieval_module: RetrievalOptimizationModule = None
        self.generation_module: GenerationIntegrationModule = None

        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")

        if not os.getenv("LLM_API_KEY") and not os.getenv("MOONSHOT_API_KEY"):
            raise ValueError("请设置 LLM_API_KEY 环境变量 (OpenAI 兼容中转站密钥)")

    # ------------------------------------------------------------------
    # 初始化 & 知识库构建
    # ------------------------------------------------------------------

    def initialize_system(self):
        """初始化各功能模块"""
        print("🚀 正在初始化 RAG 系统...")

        print("  · 数据准备模块")
        self.data_module = DataPreparationModule(self.config.data_path)

        print("  · 索引构建模块")
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path,
        )

        print("  · 生成集成模块")
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        print("✅ 系统初始化完成！\n")

    def build_knowledge_base(self):
        """加载或构建向量知识库"""
        print("📚 正在构建知识库...")

        # 文档加载与分块与索引是否持久化无关，统一执行
        self.data_module.load_documents()
        chunks = self.data_module.chunk_documents()

        vectorstore = self.index_module.load_index()
        if vectorstore is not None and not self._index_matches_chunks(vectorstore, chunks):
            logger.warning("持久化索引与当前分块不一致（路径/ID 方案变更），将重建")
            vectorstore = None

        if vectorstore is not None:
            print("  · 已加载持久化向量索引")
        else:
            print("  · 未找到可用索引，开始全新构建...")
            vectorstore = self.index_module.build_vector_store(chunks)
            self.index_module.save_index()

        self.retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)

        stats = self.data_module.get_statistics()
        print(f"\n📊 知识库统计")
        print(f"   文档总数 : {stats['total_documents']}")
        print(f"   文本块数 : {stats['total_chunks']}")
        print(f"   菜品分类 : {list(stats['categories'].keys())}")
        print(f"   难度分布 : {stats['difficulties']}")
        print("\n✅ 知识库构建完成！\n")

    # ------------------------------------------------------------------
    # 问答核心
    # ------------------------------------------------------------------

    def ask_question(self, question: str, stream: bool = False):
        """
        回答用户问题。

        Returns:
            stream=False → str；stream=True → 生成器（list 类型查询始终返回 str）
        """
        if not self.retrieval_module or not self.generation_module:
            raise ValueError("请先调用 build_knowledge_base() 构建知识库")

        print(f"\n❓ 用户问题: {question}")

        # 1. 查询路由
        route_type = self.generation_module.query_router(question)
        print(f"🎯 查询类型: {route_type}")

        # 2. 查询重写（list 类型保持原查询）
        if route_type == "list":
            rewritten_query = question
            print(f"📝 列表查询，保持原样")
        else:
            print("🤖 智能查询重写...")
            rewritten_query = self.generation_module.query_rewrite(question)

        # 3. 检索
        print("🔍 检索相关文档...")
        filters = self._extract_filters(question)
        if filters:
            print(f"   过滤条件: {filters}")
            relevant_chunks = self.retrieval_module.metadata_filtered_search(
                rewritten_query, filters, top_k=self.config.top_k
            )
        else:
            relevant_chunks = self.retrieval_module.hybrid_search(
                rewritten_query, top_k=self.config.top_k
            )

        if relevant_chunks:
            info = ", ".join(_format_chunk_info(c) for c in relevant_chunks)
            print(f"   找到 {len(relevant_chunks)} 个相关块: {info}")
        else:
            print("   未找到相关内容")
            return "抱歉，没有找到相关的食谱信息。请尝试其他菜品名称或关键词。"

        # 4. 获取父文档
        relevant_docs = self.data_module.get_parent_documents(relevant_chunks)
        names = _doc_names(relevant_docs)
        if names:
            print(f"   命中菜品: {', '.join(names)}")

        # 5. 按路由类型生成回答
        if route_type == "list":
            print("📋 生成菜品列表...")
            return self.generation_module.generate_list_answer(question, relevant_docs)

        print("✍️ 生成回答...")
        if route_type == "detail":
            return self.generation_module.generate_step_by_step_answer(
                question, relevant_docs, stream=stream
            )
        return self.generation_module.generate_basic_answer(
            question, relevant_docs, stream=stream
        )

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _index_matches_chunks(vectorstore, chunks: list) -> bool:
        """校验持久化索引与当前分块是否一致（通过 chunk_id 重叠抽样判断）。

        路径变化、分块/ID 方案变更都会导致 chunk_id 不匹配，此时必须重建索引，
        否则向量检索出的块无法反查父文档。
        """
        if vectorstore is None or not chunks:
            return False
        fresh_ids = {c.metadata.get("chunk_id") for c in chunks}

        try:
            stored_docs = list(vectorstore.docstore._dict.values())
        except AttributeError:
            return True  # 无法访问 docstore 时默认信任已加载索引

        overlap = sum(
            1 for d in stored_docs[:50]
            if d.metadata.get("chunk_id") in fresh_ids
        )
        return overlap >= 10  # 抽样 50 条中至少 10 条重叠视为兼容

    @staticmethod
    def _extract_filters(query: str) -> dict:
        """从用户问题中提取元数据过滤条件（分类 / 难度）"""
        filters = {}

        for cat in DataPreparationModule.get_supported_categories():
            if cat in query:
                filters["category"] = cat
                break

        for diff in sorted(
            DataPreparationModule.get_supported_difficulties(),
            key=len,
            reverse=True,
        ):
            if diff in query:
                # 基础难度（如"简单"）应同时匹配带"非常"前缀的难度（如"非常简单"）。
                # FAISS 对 list 条件按成员包含处理，天然支持多值匹配。
                related = [
                    d for d in DataPreparationModule.get_supported_difficulties()
                    if d == diff or d == f"非常{diff}"
                ]
                filters["difficulty"] = related if len(related) > 1 else diff
                break

        return filters

    # ------------------------------------------------------------------
    # 交互式入口
    # ------------------------------------------------------------------

    def run_interactive(self):
        """运行交互式问答循环"""
        print("=" * 60)
        print("🍽️   尝尝咸淡 RAG 系统  —  交互式问答   🍽️")
        print("=" * 60)
        print("💡 解决你的选择困难症，告别「今天吃什么」的世纪难题！")
        print("   输入 '退出' 或回车结束\n")

        self.initialize_system()
        self.build_knowledge_base()

        while True:
            try:
                user_input = input("您的问题: ").strip()
                if user_input.lower() in {"退出", "quit", "exit", ""}:
                    break

                stream_choice = input("流式输出? (y/n, 默认y): ").strip().lower()
                use_stream = stream_choice != "n"

                print("\n回答:")
                result = self.ask_question(user_input, stream=use_stream)

                # list 类型查询始终返回 str，其余类型在 stream=True 时返回生成器
                if isinstance(result, str):
                    print(f"{result}\n")
                else:
                    for chunk in result:
                        print(chunk, end="", flush=True)
                    print("\n")

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.exception("处理问题时出错")
                print(f"处理问题时出错: {e}\n")

        print("\n感谢使用尝尝咸淡 RAG 系统！")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    try:
        rag_system = RecipeRAGSystem()
        rag_system.run_interactive()
    except Exception as e:
        logger.exception("系统运行出错")
        print(f"系统错误: {e}")


if __name__ == "__main__":
    main()
