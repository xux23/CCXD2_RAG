# 尝尝咸淡 RAG 系统 · 学习文档

> 基于 [datawhalechina/all-in-rag](https://github.com/datawhalechina/all-in-rag) 第八章实战项目构建的**食谱智能问答系统**。
> 用 RAG(检索增强生成)技术解决"今天吃什么"的选择困难症:向系统提问,它从 323 份菜谱中检索相关内容,再由大模型生成回答。

---

## 目录

1. [项目简介](#1-项目简介)
2. [技术栈](#2-技术栈)
3. [环境搭建](#3-环境搭建)
4. [目录结构](#4-目录结构)
5. [系统架构与数据流](#5-系统架构与数据流)
6. [模块详解](#6-模块详解)
7. [核心技术原理](#7-核心技术原理)
8. [配置说明](#8-配置说明)
9. [使用指南](#9-使用指南)
10. [实测效果](#10-实测效果)
11. [常见问题与踩坑记录](#11-常见问题与踩坑记录)
12. [进阶扩展方向](#12-进阶扩展方向)

---

## 1. 项目简介

### 1.1 项目背景

灵感来自开源菜谱项目 [程序员做饭指南(HowToCook)](https://github.com/Anduin2017/HowToCook):300+ 份 Markdown 菜谱,结构高度规整——每道菜都有"必备原料和工具 / 计算 / 操作 / 附加内容"等固定小节,并用 ★ 表示难度。这种结构化数据非常适合做 RAG。

### 1.2 系统能力

用户可以用自然语言提问,系统支持三类问题:

| 问题类型 | 示例 | 回答方式 |
|---|---|---|
| **菜品做法** | "宫保鸡丁怎么做?" | 分步指导:介绍、食材、步骤、技巧 |
| **菜品推荐** | "推荐几个简单的素菜" | 列表式回答,自动按"素菜 + 简单"过滤 |
| **一般信息** | "红烧肉需要什么食材?" | 基础回答模式 |

### 1.3 本项目的定制点(相对原教程)

| 定制项 | 说明 |
|---|---|
| 依赖管理 | 用 **uv** 替代 conda/pip 管理虚拟环境 |
| LLM 接入 | 原代码用 `MoonshotChat`(仅支持 Kimi),改为 `ChatOpenAI` 支持**任意 OpenAI 兼容端点**(当前接阿里云百炼) |
| 向量模型 | 原代码用本地 HF 模型 `bge-small-zh-v1.5`,改为**API 向量模型** `text-embedding-v4`(阿里云百炼),无需本地下载模型 |
| 索引缓存 | 构建好的 FAISS 索引持久化在 `vector_index/`,二次启动秒级加载 |

---

## 2. 技术栈

| 类别 | 技术 |
|---|---|
| 语言 | Python 3.12.7 |
| 环境管理 | uv 0.12.x |
| LLM 框架 | LangChain 0.3.26(`langchain-core` / `langchain-community` / `langchain-openai` / `langchain-text-splitters`) |
| 向量数据库 | FAISS(`faiss-cpu`) |
| 稀疏检索 | BM25(`rank_bm25` 经 `langchain_community` 封装) |
| 向量模型 | `text-embedding-v4`(阿里云百炼 API,1024 维) |
| LLM | `deepseek-v4-flash-0731`(阿里云百炼,OpenAI 兼容) |
| 分块 | `MarkdownHeaderTextSplitter`(按 Markdown 标题层级分块) |
| 配置 | `.env` + `python-dotenv` |

### 依赖清单(`requirements.txt`)

```
langchain==0.3.26
langchain-huggingface==0.3.1
langchain-text-splitters==0.3.8
langchain-unstructured==0.1.6
langchain-community==0.3.27
faiss-cpu>=1.7.0
unstructured==0.18.11
Markdown==3.8.2
sentence-transformers>=3.0.0
lazy_loader==0.4
rank_bm25==0.2.2
openai>=1.86.0,<2.0.0
python-dotenv>=1.0.0
```

> 注:`langchain-unstructured` / `unstructured` 在 Windows 上较重,本项目代码实际未使用,安装失败可跳过。

---

## 3. 环境搭建

### 3.1 前置条件

- Python 3.12.x
- uv(或 pip + venv)
- 一个 OpenAI 兼容的 LLM API(当前为阿里云百炼)

### 3.2 创建虚拟环境并安装依赖

```powershell
cd code/C8

# 用 uv 创建 Python 3.12 虚拟环境(首次自动创建 .venv)
uv venv --python 3.12.7

# 安装依赖
uv pip install -r requirements.txt
```

> ⚠️ **uv 环境定位注意**:`uv run` 会沿目录向上查找 `pyproject.toml`/`uv.lock`,把它们所在的目录当作"项目根",虚拟环境统一放在项目根的 `.venv` 下。本项目根目录已有 `pyproject.toml` + `uv.lock`(uv 自动生成),所以无论在哪个子目录执行 `uv run`,都会使用 `D:\...\demo2\.venv`。**不要在子目录单独建 `.venv`**,否则会被忽略。

### 3.3 配置 API

复制 `.env.example` 为 `.env` 并填写:

```env
LLM_API_BASE=https://your-relay.example.com/v1
LLM_API_KEY=sk-xxx
LLM_MODEL=deepseek-v4-flash-0731
EMBEDDING_MODEL=text-embedding-v4
```

`.env` 已被 `.gitignore` 忽略,不会提交到版本库,请妥善保管其中的密钥。

### 3.4 首次启动(自动建索引)

```powershell
cd code/C8
uv run python main.py
```

首次启动没有索引缓存,会自动走完整流程:加载文档 → 分块 → 向量化 → 建 FAISS 索引 → 保存。之后启动会直接加载缓存,秒级进入问答界面。

---

## 4. 目录结构

```
demo2/                                  # 项目根 (uv 项目)
├── .venv/                              # uv 虚拟环境
├── pyproject.toml                      # uv 自动生成的项目文件
├── uv.lock                             # uv 锁文件
├── LEARNING.md                         # 本文档
├── code/C8/                            # ★ 尝尝咸淡RAG系统
│   ├── config.py                       # 配置管理 (RAGConfig)
│   ├── main.py                         # 主程序入口 (RecipeRAGSystem)
│   ├── requirements.txt                # 依赖列表
│   ├── .env                            # 本地环境变量 (密钥, 已 git 忽略)
│   ├── .env.example                    # 环境变量模板
│   ├── rag_modules/                    # 核心模块包
│   │   ├── __init__.py                 # 包导出
│   │   ├── data_preparation.py         # ① 数据准备模块
│   │   ├── index_construction.py       # ② 索引构建模块
│   │   ├── retrieval_optimization.py   # ③ 检索优化模块
│   │   └── generation_integration.py   # ④ 生成集成模块
│   └── vector_index/                   # FAISS 索引缓存 (自动生成)
│       ├── index.faiss                 # 向量数据
│       └── index.pkl                   # 文档元数据
└── data/C8/cook/dishes/                # 菜谱数据 (323 份 .md)
    ├── meat_dish/                      # 荤菜
    ├── vegetable_dish/                 # 素菜
    ├── soup/                           # 汤品
    ├── dessert/                        # 甜品
    ├── breakfast/                      # 早餐
    ├── staple/                         # 主食
    ├── aquatic/                        # 水产
    ├── condiment/                      # 调料
    ├── drink/                          # 饮品
    ├── semi-finished/                  # 半成品
    └── template/                       # 模板
```

---

## 5. 系统架构与数据流

### 5.1 总体流程

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│  数据准备    │ →  │   索引构建    │ →  │   检索优化   │ →  │   生成集成    │
│ 加载/分块/  │    │ 向量化/FAISS │    │ 混合检索/  │    │ 路由/重写/   │
│ 元数据增强  │    │ 持久化       │    │ RRF重排     │    │ 回答生成     │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
```

### 5.2 一次问答的完整数据流(对应 `main.py` 的 `ask_question`)

```
用户输入
   │
   ▼
① 查询路由 (query_router)          → list / detail / general 三类
   │
   ▼
② 查询重写 (query_rewrite)         → list 保持原样;其余交给 LLM 判断是否重写
   │
   ▼
③ 元数据过滤条件提取                → 从问题中自动识别 分类/难度 关键词
   │
   ▼
④ 混合检索 (hybrid_search)         → 向量检索 (top5) + BM25 检索 (top5)
   │                                → RRF 融合重排 → 取 top_k=3 个子块
   ▼
⑤ 父子文档映射 (get_parent_documents) → 由子块找回完整菜谱, 按命中次数排序去重
   │
   ▼
⑥ 生成回答 (按路由类型)
      list    → generate_list_answer (直接列菜名, 不调 LLM)
      detail  → generate_step_by_step_answer (分步指导 prompt)
      general → generate_basic_answer (通用回答 prompt)
```

### 5.3 启动时的索引缓存逻辑(`build_knowledge_base`)

```
检查 vector_index/ 是否存在
   ├── 存在 → load_index() 秒级加载
   │          (但仍需加载文档+分块, 供检索模块和父子映射使用)
   └── 不存在 → 加载文档 → 分块 → build_vector_index() → save_index()
```

---

## 6. 模块详解

### 6.1 `config.py` — 配置管理

用 `@dataclass` 定义 `RAGConfig`,集中管理系统所有参数:

| 字段 | 默认值 | 说明 |
|---|---|---|
| `data_path` | `../../data/C8/cook` | 菜谱数据目录 |
| `index_save_path` | `./vector_index` | FAISS 索引保存路径 |
| `embedding_model` | `os.getenv("EMBEDDING_MODEL", "text-embedding-v4")` | 向量模型,优先读环境变量 |
| `llm_model` | `os.getenv("LLM_MODEL", "kimi-k2-0711-preview")` | 生成模型 |
| `top_k` | `3` | 检索返回块数 |
| `temperature` | `0.1` | 生成温度(低=更稳定) |
| `max_tokens` | `2048` | 回答最大 token 数 |

关键点:
- **环境变量在 dataclass 字段默认值处读取**,因此 `main.py` 里 `load_dotenv()` 必须在 `from config import ...` **之前**执行,否则 `.env` 中的模型名读不到(本项目已修正此顺序)。
- 提供 `from_dict` / `to_dict` 便于序列化扩展。

### 6.2 `rag_modules/data_preparation.py` — 数据准备模块

职责:**加载菜谱 → 元数据增强 → 结构分块 → 建立父子文档关系**。

#### (1) `load_documents()` — 加载文档

- 递归遍历 `data_path` 下所有 `*.md`,以 UTF-8 读入,保留原始 Markdown 格式(不做清洗,因为数据本身规整)。
- 为每份菜谱生成**确定性父文档 ID**:`hashlib.md5(相对路径)`。同一路径每次运行 ID 一致,保证索引与文档可稳定对应。
- 元数据初始为:`source`(绝对路径)、`parent_id`、`doc_type="parent"`。

#### (2) `_enhance_metadata()` — 元数据增强

根据**文件路径**和**文档内容**自动补齐三类元数据:

```python
# ① 分类: 从路径中的目录名映射中文标签
CATEGORY_MAPPING = {
    'meat_dish': '荤菜', 'vegetable_dish': '素菜', 'soup': '汤品',
    'dessert': '甜品', 'breakfast': '早餐', 'staple': '主食',
    'aquatic': '水产', 'condiment': '调料', 'drink': '饮品'
}
# ② 菜名: 取文件名 stem, 如 西红柿炒鸡蛋.md → "西红柿炒鸡蛋"
# ③ 难度: 用正则 \★+ 匹配内容中的星号个数
#    ★=非常简单, ★★=简单, ★★★=中等, ★★★★=困难, ★★★★★=非常困难
```

#### (3) `chunk_documents()` → `_markdown_header_split()` — Markdown 结构感知分块

这是本项目分块的核心策略。使用 `MarkdownHeaderTextSplitter` 按标题层级切分:

```python
headers_to_split_on = [
    ("#", "主标题"),      # 菜品名称
    ("##", "二级标题"),   # 必备原料、计算、操作等
    ("###", "三级标题"),  # 简易版本、复杂版本等
]
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False   # 保留标题, 便于理解上下文
)
```

- 每个子块获得:`chunk_id`(uuid)、`parent_id`(继承父文档)、`doc_type="child"`、`chunk_index`、`batch_index`、`chunk_size`。
- 同时维护 `parent_child_map: {child_id → parent_id}`,为"小块检索、大块生成"打基础。
- 分块失败(无标题结构)的文档,整篇作为一个 chunk 兜底。

#### (4) `get_parent_documents()` — 父子文档映射(智能去重)

```python
输入: 检索到的子块列表
逻辑: 统计每个 parent_id 被命中的次数 (相关性指标)
     → 按命中次数降序排序 → 返回去重后的完整父文档列表
输出: 完整菜谱文档 (按相关性排序)
```

**设计思想**:小块的精确性用于检索,大块的完整性用于生成(详见 [7.2](#72-父子文档小块检索大块生成))。

#### (5) 辅助方法

- `get_supported_categories()` / `get_supported_difficulties()`:供查询过滤复用。
- `filter_documents_by_category/difficulty()`:按元数据过滤文档。
- `get_statistics()`:统计文档数、分块数、分类分布、难度分布、平均块大小。
- `export_metadata()`:把元数据导出为 JSON(便于分析数据质量)。

### 6.3 `rag_modules/index_construction.py` — 索引构建模块

职责:**初始化嵌入模型 → 构建 FAISS 向量索引 → 持久化/加载**。

#### (1) `setup_embeddings()` — 双模式嵌入模型

这是本项目针对 API 化的核心改动,支持两种嵌入来源:

```python
# 模式 A: 本地 HuggingFace 模型 (模型名以 BAAI/ sentence-transformers/ moka-ai/ 开头)
if self.model_name.startswith(("BAAI/", "sentence-transformers/", "moka-ai/")):
    self.embeddings = HuggingFaceEmbeddings(
        model_name=self.model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}   # 归一化, 便于余弦相似度
    )
# 模式 B: OpenAI 兼容 API 端点 (当前使用)
elif api_base and api_key:
    self.embeddings = OpenAIEmbeddings(
        model=self.model_name,
        api_key=api_key,
        base_url=api_base,
        chunk_size=10,                     # 阿里云百炼单批上限 10
        check_embedding_ctx_length=False,  # 百炼不支持 tokenize 输入, 直接发文本
    )
```

> 两个关键适配参数(踩坑点,见 [11.4](#114-阿里云百炼-embedding-400-错误)):
> - `check_embedding_ctx_length=False`:默认开启时 langchain 会把文本 tokenize 成 token ID 数组发送,百炼拒绝这种格式,必须关闭。
> - `chunk_size=10`:百炼 `text-embedding-v4` 单次请求最多 10 条文本,超过报 400。

#### (2) FAISS 索引操作

```python
# 构建: 一次性把全部 chunks 向量化并入库
self.vectorstore = FAISS.from_documents(chunks, embedding=self.embeddings)

# 保存: 生成 index.faiss (向量) + index.pkl (元数据)
self.vectorstore.save_local(self.index_save_path)

# 加载: 反序列化时需显式允许 (官方安全要求)
FAISS.load_local(self.index_save_path, self.embeddings,
                 allow_dangerous_deserialization=True)
```

### 6.4 `rag_modules/retrieval_optimization.py` — 检索优化模块

职责:**混合检索(向量 + BM25)+ RRF 融合重排 + 元数据过滤**。

#### (1) `setup_retrievers()` — 双路检索器

```python
# 向量检索器: FAISS 相似度检索, 取 top 5
self.vector_retriever = self.vectorstore.as_retriever(
    search_type="similarity", search_kwargs={"k": 5})

# 稀疏检索器: BM25 关键词检索, 取 top 5
self.bm25_retriever = BM25Retriever.from_documents(self.chunks, k=5)
```

**为什么混合检索?** 向量检索擅长语义相似(同义改写也能命中),但精确关键词(菜名、食材名)可能被语义漂移带偏;BM25 精确匹配关键词,但无法理解语义。二者互补。

#### (2) `hybrid_search()` + `_rrf_rerank()` — RRF 融合

```python
RRF 公式: score(doc) = Σ 1 / (k + rank)     # k 通常取 60

# 例: 某文档在向量检索排第 2, 在 BM25 排第 5
# score = 1/(60+2) + 1/(60+5) = 0.01613 + 0.01538 = 0.03151
```

- 用 `hashlib.md5(page_content)` 作为文档去重标识(同一内容在两路检索中都出现时分数累加)。
- 同时出现在两路结果中的文档得分更高,自然排在前面——这就是 RRF 的价值:融合两路信号,并优先两路都认可的文档。
- 最终取 `top_k=3`(由 `config.top_k` 控制)。

#### (3) `metadata_filtered_search()` — 元数据过滤检索

```python
流程: 先混合检索 top_k*3 个候选 → 逐个检查元数据是否满足 filters
      (如 category=素菜, difficulty=简单) → 取前 top_k 个
用途: "推荐几个简单的素菜" 这类问题, 先检索再按元数据精确过滤
```

### 6.5 `rag_modules/generation_integration.py` — 生成集成模块

职责:**LLM 接入 + 查询路由 + 查询重写 + 三种回答模式(含流式)**。

#### (1) `setup_llm()` — 通用 OpenAI 兼容接入

```python
self.llm = ChatOpenAI(
    model=self.model_name,
    temperature=self.temperature,
    max_tokens=self.max_tokens,
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_API_BASE"),
)
```

> 原教程用 `MoonshotChat`(base_url 写死 Kimi),本项目改为 `ChatOpenAI`,base_url 从环境变量读取,因此可对接任意 OpenAI 兼容端点(阿里云百炼、DeepSeek、OpenRouter 等)。旧的 `MOONSHOT_API_KEY` 仍作为兼容回退。

#### (2) `query_router()` — 查询路由

用 LLM 把问题分成三类,并做**白名单校验**(防止 LLM 输出乱七八糟的内容):

```python
list    → 想要推荐/列表, 只要菜名   "推荐几个素菜"
detail  → 要具体做法/信息           "宫保鸡丁怎么做"
general → 其他一般性问题            "什么是川菜"
结果不在白名单 → 默认归为 general
```

#### (3) `query_rewrite()` — 查询重写

用 LLM 判断问题是否需要重写以提高检索效果:
- **具体明确的问题**(含菜名、明确步骤)→ 保持原样,不浪费一次调用。
- **模糊宽泛的问题**("做菜"、"推荐个菜")→ 重写成带烹饪术语的检索词,如 `"做菜" → "简单易做的家常菜谱"`。
- 重写结果与原文不同时记日志,便于追踪。

#### (4) 三种回答模式

| 方法 | 触发路由 | 说明 |
|---|---|---|
| `generate_list_answer` | list | **不调 LLM**,直接从元数据提取菜名拼列表(快、省 token) |
| `generate_step_by_step_answer` | detail | 分步指导 prompt:菜品介绍 / 所需食材 / 制作步骤 / 制作技巧 |
| `generate_basic_answer` | general | 通用烹饪助手 prompt,信息不足时诚实说明 |

每种都有对应的流式版本(`*_stream`),用 `chain.stream()` 逐 token 输出。

#### (5) `_build_context()` — 上下文组装

把父文档按格式拼装,带元数据标签,并限制总长度:

```text
【食谱 1】 宫保鸡丁 | 分类: 荤菜 | 难度: 困难
<完整菜谱内容>
==================================================
【食谱 2】 ...
```

---

## 7. 核心技术原理

### 7.1 RAG 基本概念

RAG = **Retrieval-Augmented Generation(检索增强生成)**。为什么不直接问 LLM?

- LLM 的知识有截止日期,不知道你的私有/最新数据;
- LLM 会"幻觉",编造不存在的菜谱。

RAG 的解决方式:先把文档切块向量化存起来;用户提问时,检索出最相关的块,连同问题一起喂给 LLM,让它"看着资料回答"。**回答有据可依,准确率大幅提升**。

### 7.2 父子文档:"小块检索,大块生成"

这是本项目最有学习价值的设计,解决了一个经典矛盾:

```
问题: 菜谱整篇约 700 字, 直接整篇做块 →
      "宫保鸡丁需要什么调料" 这类具体问题在整篇里占比太小, 向量检索排名靠后
问题: 严格按标题切成小块 (操作/原料/计算...) →
      检索精确了, 但可能只找到"操作"块, 缺少"必备原料和工具", 回答不完整

方案: 父文档 = 完整菜谱 (用于生成, 上下文完整)
     子文档 = 按标题切的小块 (用于检索, 精确命中)
     检索子块 → 通过 parent_id 找回父文档 → 把完整菜谱交给 LLM
```

一句话:**用小块的精确性找到相关内容,用大块的完整性保证回答质量**。

### 7.3 混合检索 + RRF 重排

- **稠密检索(向量)**:文本 → 语义向量 → 余弦相似度。理解"意思",但可能抓不住精确的菜名。
- **稀疏检索(BM25)**:词频统计 + 逆文档频率。精确匹配关键词,但不懂语义。
- **RRF 融合**:不看分数绝对值(两路分数不可比),只看**排名**,`1/(k+rank)` 求和。简单、鲁棒、无需调权重。

### 7.4 查询路由 + 查询重写 + 元数据过滤 —— "查询理解三件套"

| 技术 | 作用 | 成本 |
|---|---|---|
| 查询路由 | 决定回答形式(list/detail/general) | 1 次 LLM 调用 |
| 查询重写 | 把模糊问题变成利于检索的词 | 1 次 LLM 调用(detail/general 时) |
| 元数据过滤 | 从问题提取"素菜""简单"等条件,检索后精确过滤 | 0 次 LLM 调用,纯规则 |

三者叠加,让同一个检索/生成管道能应对风格迥异的用户问题。

### 7.5 索引持久化缓存

FAISS 索引(向量 + 元数据)保存到 `vector_index/`:
- **首次构建**:323 文档 → 1764 块 → 1764 条向量,API 向量化约需几分钟。
- **二次启动**:直接反序列化加载,秒级进入问答。
- 换 embedding 模型后必须删掉旧索引重建(向量维度/语义空间都变了)。

### 7.6 向量模型选择的考量

| 方案 | 优点 | 缺点 |
|---|---|---|
| 本地 HF 模型(`bge-small-zh-v1.5`) | 免费、离线、无网络依赖 | 需下载模型(国内网络慢)、占内存 |
| API 向量模型(`text-embedding-v4`) | 免下载、效果好、按量付费 | 每次构建/查询都走网络、有配额限制 |

本项目当前用 API 模式;若想切回本地,把 `EMBEDDING_MODEL` 改成 `BAAI/bge-small-zh-v1.5` 并删除 `vector_index/` 重建即可。

---

## 8. 配置说明

### 8.1 环境变量(`.env`)

| 变量 | 必填 | 说明 |
|---|---|---|
| `LLM_API_BASE` | ✅ | OpenAI 兼容端点地址,如 `https://xxx.com/compatible-mode/v1` |
| `LLM_API_KEY` | ✅ | API 密钥 |
| `LLM_MODEL` | ✅ | 生成模型名 |
| `EMBEDDING_MODEL` | ✅ | 向量模型名(API 模式)或本地 HF 模型名 |
| `HF_ENDPOINT` | ❌ | HuggingFace 镜像(本地模型下载失败时用 `https://hf-mirror.com`) |

### 8.2 `RAGConfig` 可调参数

修改 `config.py` 中的 `RAGConfig` 即可:

```python
top_k = 3          # 检索块数: 越大上下文越全, 但更慢更贵
temperature = 0.1  # 生成温度: 调菜谱场景要稳定, 建议保持低值
max_tokens = 2048  # 回答长度上限
```

---

## 9. 使用指南

### 9.1 启动交互式问答

```powershell
cd code/C8
uv run python main.py
```

```
============================================================
🍽️  尝尝咸淡RAG系统 - 交互式问答  🍽️
============================================================
💡 解决您的选择困难症，告别'今天吃什么'的世纪难题！
```

### 9.2 问答示例

```
您的问题: 宫保鸡丁怎么做
是否使用流式输出? (y/n, 默认y):
回答: ...
```

支持的退出词:`退出` / `quit` / `exit` / 空回车。

### 9.3 常见提问模板

| 意图 | 问法 |
|---|---|
| 做法 | "蛋炒饭怎么做"、"糖醋排骨的制作步骤" |
| 食材 | "红烧肉需要什么食材" |
| 推荐 | "推荐几个简单的素菜"、"有什么川菜" |
| 技巧 | "如何炒菜不粘锅" |

---

## 10. 实测效果

### 10.1 知识库统计(当前数据)

```
文档总数: 323
文本块数: 1764
菜品分类: 水产 24 / 早餐 22 / 调料 9 / 饮品 21 / 荤菜 97 / 其他 11
         汤品 21 / 主食 47 / 素菜 54 / 甜品 17
难度分布: 非常困难 20 / 困难 78 / 中等 115 / 简单 83 / 非常简单 27
```

### 10.2 混合检索效果(`宫保鸡丁怎么做`)

```
1. [宫保鸡丁]  # 宫保鸡丁的做法 (rrf=0.0164)
2. [鸡蛋羹]    ## 附加内容  (rrf=0.0164)
3. [宫保鸡丁]  ## 操作      (rrf=0.0161)
```

目标菜谱稳定出现在前排;父子映射后拿到完整《宫保鸡丁》文档。

### 10.3 端到端回答质量(节选)

**问:宫保鸡丁怎么做**(路由 detail)→ 生成了:菜品介绍(糊辣荔枝味)、食材清单(主料/腌料/碗芡分列)、8 步制作流程(家常版+进阶版)、4 条制作技巧,信息全部源自检索到的真实菜谱。

**问:推荐几个简单的素菜**(路由 list)→ 自动提取过滤条件 `{'category': '素菜', 'difficulty': '简单'}`,检索后推荐:清炒花菜、鸡蛋羹、蚝油生菜。

---

## 11. 常见问题与踩坑记录

### 11.1 `uv run` 找不到已安装的包

**症状**:`ModuleNotFoundError: No module named 'langchain_text_splitters'`,但 `.venv` 里明明有。

**原因**:`uv run` 沿目录向上找 `pyproject.toml`,把 git 根当项目根,虚拟环境放 `demo2/.venv`;若你在 `code/C8/.venv` 又建了一个,uv 不认它,会新建/使用根的空环境。

**解决**:环境统一放项目根 `.venv`;确保根目录有 `pyproject.toml`(uv 会自动生成)。

### 11.2 PowerShell 下 `uv`/python 报 "exit status 1" 但实际成功

**原因**:uv/python 把日志写到 stderr,PowerShell 把 stderr 输出当作错误(NativeCommandError),导致命令"看起来失败"。

**判断方法**:看输出内容是否包含 Traceback / Error;若只是 INFO 日志 + exit 1,属误报。可加 `2>$null` 抑制 stderr 再判断。

### 11.3 `faiss.swigfaiss_avx2` ModuleNotFoundError

**症状**:`Loading faiss` 时打印 `Could not load library with AVX2 support...`。

**原因**:faiss-cpu 尝试加载 AVX2 加速库失败,自动回退到普通实现。

**结论**:非致命警告,功能正常(日志里会接着显示 `Successfully loaded faiss`),可忽略。

### 11.4 阿里云百炼 Embedding 400 错误

**坑 1**:`Field required: input.contents`
- 原因:langchain `OpenAIEmbeddings` 默认 `check_embedding_ctx_length=True`,会 tokenize 成 token 数组发送,百炼不认。
- 解决:`check_embedding_ctx_length=False`,直接发送原始文本。

**坑 2**:`batch size is invalid, it should not be larger than 10`
- 原因:百炼 `text-embedding-v4` 单批上限 10 条。
- 解决:`chunk_size=10`(langchain 会按批切分)。

### 11.5 换 embedding 模型后检索异常

FAISS 索引与 embedding 模型强绑定(维度、语义空间)。换模型必须:

```powershell
Remove-Item vector_index -Recurse -Force
uv run python main.py   # 自动重建
```

### 11.6 首次启动 HuggingFace 下载超时(本地模型模式)

在国内网络环境,`huggingface.co` 可能连不上。在 `.env` 加:

```env
HF_ENDPOINT=https://hf-mirror.com
```

或改用 API 向量模型(本项目当前方案,无此问题)。

### 11.7 LLM 回答为空

排查:模型是否支持 `max_tokens` 字段、是否有 reasoning tokens(如 deepseek-v4-flash 的 reasoning 会吃掉 max_tokens 预算)、`temperature` 是否被端点支持。可用 `max_tokens=2048` + 增大预算或换兼容参数。

---

## 12. 进阶扩展方向

1. **评估体系**:用 RAGAS / LangSmith 评估检索命中率与回答忠实度(faithfulness),量化"小块检索大块生成"到底比整篇分块好多少。
2. **重排模型**:RRF 之后再接一个 cross-encoder reranker(如 `bge-reranker`),进一步提升 top-k 精度。
3. **多路召回增强**:加入"菜名倒排索引"作为第三路召回,菜名类问题会命中更准。
4. **流式 + 引用溯源**:回答中标注"来自哪份菜谱",提升可信度。
5. **前端化**:把交互式 CLI 包成 Gradio / FastAPI 服务,支持网页问答。
6. **Graph RAG**:参考教程第九章,把菜谱实体(菜名/食材/工具)建知识图谱,回答"有哪些菜用到豆腐"这类关联问题。

---

## 附:各模块职责速查

| 文件 | 类 | 一句话职责 |
|---|---|---|
| `config.py` | `RAGConfig` | 集中管理全部配置 |
| `main.py` | `RecipeRAGSystem` | 编排四个模块, 处理问答主流程 |
| `data_preparation.py` | `DataPreparationModule` | 加载菜谱 + 元数据 + 结构分块 + 父子映射 |
| `index_construction.py` | `IndexConstructionModule` | 嵌入向量化 + FAISS 构建/持久化 |
| `retrieval_optimization.py` | `RetrievalOptimizationModule` | 混合检索 + RRF 重排 + 元数据过滤 |
| `generation_integration.py` | `GenerationIntegrationModule` | LLM 接入 + 路由/重写 + 三种回答模式 |
