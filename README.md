# 尝尝咸淡 · RAG 食谱问答系统

用 RAG（检索增强生成）解决"今天吃什么"的选择困难症：用户用自然语言提问，系统从 **323 篇结构化菜谱**中检索相关内容，再由大模型生成回答。

菜谱数据来自开源项目 [程序员做饭指南 HowToCook](https://github.com/Anduin2017/HowToCook)——每道菜都有"必备原料和工具 / 计算 / 操作 / 附加内容"的固定小节，并用 ★ 标注难度，这种高度规整的结构非常适合做检索。

## 能回答的三类问题

| 问题类型 | 示例 | 回答方式 |
|---|---|---|
| 菜品做法 | "宫保鸡丁怎么做？" | 分步指导：介绍、食材、步骤、技巧 |
| 菜品推荐 | "推荐几个简单的素菜" | 列表式回答，按"素菜 + 简单"自动过滤 |
| 一般信息 | "红烧肉需要什么食材？" | 基础问答模式 |

## 技术方案

四个模块串成一条链路，互相独立、可单独测试：

```
data_preparation  →  index_construction  →  retrieval_optimization  →  generation_integration
   数据准备              索引构建                检索优化                    生成集成
```

1. **数据准备（父文档 + 子块）**：按 Markdown 标题层级把每道菜切成小块，同时保留完整菜谱作为父文档。
   检索时用小块匹配（语义更聚焦），命中后通过 `parent_id` 回填完整菜谱，避免只拿到半截内容。
2. **索引构建（双索引）**：FAISS 向量索引 + BM25 关键词索引，两种索引共用同一份 chunk。
3. **检索优化（混合召回 + RRF 融合）**：向量检索取 top 5、BM25 检索取 top 5，各自排名经 **RRF（Reciprocal Rank Fusion）**
   重排后取最终 top 3；支持按菜品分类、难度做元数据过滤。
4. **生成集成（按问题类型路由）**：先判断问题属于哪一类——推荐类问题直接用代码提取菜名、**不调用大模型**，
   既能保证筛选准确，也减少了接口调用次数。

## 目录结构

```
.
├── main.py                        # 入口：RecipeRAGSystem 主类 + 命令行交互
├── config.py                      # 配置 dataclass（路径 / 模型 / 检索 / 生成参数）
├── rag_modules/
│   ├── data_preparation.py        # 数据准备：切分、父文档与子块
│   ├── index_construction.py      # 索引构建：FAISS + BM25
│   ├── retrieval_optimization.py  # 检索优化：混合召回 + RRF + 元数据过滤
│   └── generation_integration.py  # 生成集成：问题分类路由 + 提示词
├── data/C8/cook/dishes/           # 323 篇菜谱（含配图）
├── LEARNING.md                    # 12 章学习文档：架构、原理、踩坑记录
└── MINDMAP.md                     # 知识导图
```

## 快速开始

```bash
# 1. 配置（OpenAI 兼容端点即可，模型名按中转站支持的填）
cp .env.example .env
#   编辑 .env：LLM_API_BASE / LLM_API_KEY / LLM_MODEL / EMBEDDING_MODEL

# 2. 安装依赖（项目提供 uv.lock）
uv sync

# 3. 运行
uv run python main.py
```

可选：国内下载 Embedding 模型慢时，在 `.env` 中取消注释 `HF_ENDPOINT=https://hf-mirror.com`。

## 关键参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `top_k` | 3 | RRF 融合后最终返回的文档数 |
| 向量检索 / BM25 召回数 | 各 5 | 融合前的候选池 |
| `temperature` | 0.1 | 生成温度，菜谱类问答需要稳定输出 |
| `max_tokens` | 2048 | 单次生成上限 |

## 文档

- `LEARNING.md`：从 RAG 原理到本项目实现的完整学习笔记，含实测效果与踩坑记录
- `MINDMAP.md`：项目知识导图

## 参考

项目基于 [datawhalechina/all-in-rag](https://github.com/datawhalechina/all-in-rag) 第八章实战项目构建，
在其基础上完成了代码重构、模块拆分与检索参数调试，学习过程中的问题与解决记录见 `LEARNING.md`。
