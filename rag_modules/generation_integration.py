"""
生成集成模块 — 负责 LLM 接入与回答生成
"""

import logging
import os
from typing import Iterator, List, Union

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt 模板（集中管理，避免重复）
# ---------------------------------------------------------------------------

BASIC_ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """你是一位专业的烹饪助手。请根据以下食谱信息回答用户的问题。

用户问题: {question}

相关食谱信息:
{context}

请提供详细、实用的回答。如果信息不足，请诚实说明。

回答:"""
)

STEP_BY_STEP_PROMPT = ChatPromptTemplate.from_template(
    """你是一位专业的烹饪导师。请根据食谱信息，为用户提供详细的分步骤指导。

用户问题: {question}

相关食谱信息:
{context}

请灵活组织回答，建议包含以下部分（可根据实际内容调整）：

## 🥘 菜品介绍
[简要介绍菜品特点和难度]

## 🛒 所需食材
[列出主要食材和用量]

## 👨‍🍳 制作步骤
[详细的分步骤说明，每步包含具体操作和大概所需时间]

## 💡 制作技巧
[仅在有实用技巧时包含。优先使用原文中的实用技巧，如果原文的"附加内容"与烹饪无关或为空，可以基于制作步骤总结关键要点，或者完全省略此部分]

注意：
- 根据实际内容灵活调整结构
- 不要强行填充无关内容或重复制作步骤中的信息
- 重点突出实用性和可操作性
- 如果没有额外的技巧要分享，可以省略制作技巧部分

回答:"""
)

QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """你是一个智能查询分析助手。请分析用户的查询，判断是否需要重写以提高食谱搜索效果。

原始查询: {query}

分析规则：
1. **具体明确的查询**（直接返回原查询）：
   - 包含具体菜品名称：如"宫保鸡丁怎么做"、"红烧肉的制作方法"
   - 明确的制作询问：如"蛋炒饭需要什么食材"、"糖醋排骨的步骤"
   - 具体的烹饪技巧：如"如何炒菜不粘锅"、"怎样调制糖醋汁"

2. **模糊不清的查询**（需要重写）：
   - 过于宽泛：如"做菜"、"有什么好吃的"、"推荐个菜"
   - 缺乏具体信息：如"川菜"、"素菜"、"简单的"
   - 口语化表达：如"想吃点什么"、"有饮品推荐吗"

重写原则：
- 保持原意不变
- 增加相关烹饪术语
- 优先推荐简单易做的
- 保持简洁性

示例：
- "做菜" → "简单易做的家常菜谱"
- "有饮品推荐吗" → "简单饮品制作方法"
- "推荐个菜" → "简单家常菜推荐"
- "川菜" → "经典川菜菜谱"
- "宫保鸡丁怎么做" → "宫保鸡丁怎么做"（保持原查询）
- "红烧肉需要什么食材" → "红烧肉需要什么食材"（保持原查询）

请输出最终查询（如果不需要重写就返回原查询）:"""
)

QUERY_ROUTER_PROMPT = ChatPromptTemplate.from_template(
    """根据用户的问题，将其分类为以下三种类型之一：

1. 'list' - 用户想要获取菜品列表或推荐，只需要菜名
   例如：推荐几个素菜、有什么川菜、给我3个简单的菜

2. 'detail' - 用户想要具体的制作方法或详细信息
   例如：宫保鸡丁怎么做、制作步骤、需要什么食材

3. 'general' - 其他一般性问题
   例如：什么是川菜、制作技巧、营养价值

请只返回分类结果：list、detail 或 general

用户问题: {query}

分类结果:"""
)


# ---------------------------------------------------------------------------
# 模块主体
# ---------------------------------------------------------------------------

class GenerationIntegrationModule:
    """生成集成模块 — 负责 LLM 接入与回答生成"""

    def __init__(
        self,
        model_name: str = "qwen-plus",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None
        self._setup_llm()

    # ------------------------------------------------------------------
    # 内部：LLM 初始化
    # ------------------------------------------------------------------

    def _setup_llm(self):
        """初始化大语言模型（OpenAI 兼容中转站）"""
        api_key = os.getenv("LLM_API_KEY") or os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            raise ValueError("请设置 LLM_API_KEY 环境变量 (OpenAI 兼容中转站密钥)")

        api_base = os.getenv("LLM_API_BASE")
        if not api_base:
            raise ValueError("请设置 LLM_API_BASE 环境变量 (OpenAI 兼容中转站地址)")

        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=api_key,
            base_url=api_base,
        )
        logger.info("LLM 初始化完成: %s", self.model_name)

    # ------------------------------------------------------------------
    # 内部：上下文构建 & 链构建
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context(docs: List[Document], max_length: int = 4000) -> str:
        """将文档列表格式化为上下文字符串"""
        if not docs:
            return "暂无相关食谱信息。"

        context_parts = []
        current_length = 0

        for i, doc in enumerate(docs, 1):
            meta_info = f"【食谱 {i}】"
            if "dish_name" in doc.metadata:
                meta_info += f" {doc.metadata['dish_name']}"
            if "category" in doc.metadata:
                meta_info += f" | 分类: {doc.metadata['category']}"
            if "difficulty" in doc.metadata:
                meta_info += f" | 难度: {doc.metadata['difficulty']}"

            doc_text = f"{meta_info}\n{doc.page_content}\n"
            if current_length + len(doc_text) > max_length:
                break
            context_parts.append(doc_text)
            current_length += len(doc_text)

        divider = "\n" + "=" * 50 + "\n"
        return divider + divider.join(context_parts)

    def _build_chain(self, prompt: ChatPromptTemplate, context: str):
        """构建通用 LCEL 链（question → prompt → llm → parser）"""
        return (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def _invoke(self, prompt: ChatPromptTemplate, query: str, context: str) -> str:
        """同步调用链，返回完整字符串"""
        chain = self._build_chain(prompt, context)
        return chain.invoke(query)

    def _stream(self, prompt: ChatPromptTemplate, query: str, context: str):
        """流式调用链，yield 文本片段"""
        chain = self._build_chain(prompt, context)
        yield from chain.stream(query)

    # ------------------------------------------------------------------
    # 公开 API：查询预处理
    # ------------------------------------------------------------------

    def query_rewrite(self, query: str) -> str:
        """智能查询重写：模糊查询补充关键词，具体查询保持不变"""
        chain = (
            {"query": RunnablePassthrough()}
            | QUERY_REWRITE_PROMPT
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(query).strip()

    def query_router(self, query: str) -> str:
        """查询路由：将问题分类为 list / detail / general"""
        chain = (
            {"query": RunnablePassthrough()}
            | QUERY_ROUTER_PROMPT
            | self.llm
            | StrOutputParser()
        )
        result = chain.invoke(query).strip().lower()
        # 容错：LLM 可能输出带引号/标点/多余文字，按关键词匹配
        for keyword in ("list", "detail", "general"):
            if keyword in result:
                return keyword
        logger.warning("无法识别路由类型: %r，回退为 general", result)
        return "general"

    # ------------------------------------------------------------------
    # 公开 API：回答生成（统一 stream 参数，消除重复）
    # ------------------------------------------------------------------

    def generate_basic_answer(
        self, query: str, context_docs: List[Document], stream: bool = False
    ) -> Union[str, Iterator[str]]:
        """生成基础回答（适用于 general 类型查询）"""
        context = self._build_context(context_docs)
        if stream:
            return self._stream(BASIC_ANSWER_PROMPT, query, context)
        return self._invoke(BASIC_ANSWER_PROMPT, query, context)

    def generate_step_by_step_answer(
        self, query: str, context_docs: List[Document], stream: bool = False
    ) -> Union[str, Iterator[str]]:
        """生成分步骤详细回答（适用于 detail 类型查询）"""
        context = self._build_context(context_docs)
        if stream:
            return self._stream(STEP_BY_STEP_PROMPT, query, context)
        return self._invoke(STEP_BY_STEP_PROMPT, query, context)

    def generate_list_answer(self, query: str, context_docs: List[Document]) -> str:
        """生成菜品列表（适用于 list 类型查询，不调用 LLM）"""
        if not context_docs:
            return "没有相关菜品信息。"

        seen = set()
        dish_names = []
        for doc in context_docs:
            name = doc.metadata.get("dish_name", "未知菜品")
            if name not in seen:
                seen.add(name)
                dish_names.append(name)

        if not dish_names:
            return "没有相关菜品信息。"
        if len(dish_names) == 1:
            return f"为您推荐：{dish_names[0]}"
        if len(dish_names) <= 3:
            lines = "\n".join(f"{i+1}. {name}" for i, name in enumerate(dish_names))
            return f"为您推荐以下菜品：\n{lines}"

        top3 = "\n".join(f"{i+1}. {name}" for i, name in enumerate(dish_names[:3]))
        return f"为您推荐以下菜品：\n{top3}\n\n还有其他 {len(dish_names) - 3} 道菜品可供选择。"
