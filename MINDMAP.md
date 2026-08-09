# 尝尝咸淡 RAG 系统 — 思维导图

```
尝尝咸淡 RAG 系统(食谱问答)
│
├── 📂 数据层 data/C8/cook
│   ├── 322 篇 Markdown 菜谱
│   ├── 9 个分类: 荤菜/素菜/汤品/甜品/早餐/主食/水产/调料/饮品/半成品
│   ├── 难度: ★~★★★★★ → 非常简单/简单/中等/困难/非常困难
│   └── 每篇含: 原料/步骤/技巧 + 可选成品图
│
├── 📂 代码层（项目根目录）
│   ├── main.py — 系统编排 + 交互式入口
│   │   ├── RecipeRAGSystem
│   │   │   ├── initialize_system() 初始化 4 模块
│   │   │   ├── build_knowledge_base() 加载/构建索引
│   │   │   │   └── 索引兼容性校验(防静默失效)
│   │   │   ├── ask_question() 问答主流程
│   │   │   │   ├── query_router → list/detail/general
│   │   │   │   ├── query_rewrite 模糊查询改写
│   │   │   │   ├── _extract_filters 分类/难度过滤
│   │   │   │   ├── hybrid_search 混合检索
│   │   │   │   └── get_parent_documents 父文档反查
│   │   │   └── run_interactive() 命令行交互
│   │   └── 流式输出兼容 str/生成器
│   │
│   ├── config.py — 配置
│   │   ├── 绝对路径(基于 __file__)
│   │   ├── 模型: qwen-plus / text-embedding-v3
│   │   └── 检索: top_k=3, temperature=0.1
│   │
│   └── rag_modules/ — 四大功能模块
│       ├── data_preparation.py — 数据准备
│       │   ├── load_documents: rglob 扫描 .md + 跳过 template
│       │   ├── 元数据增强: 分类/菜名/难度(★)
│       │   ├── Markdown 标题分块 (#/##/###)
│       │   ├── 确定性 ID: md5(路径) / md5(父ID:序号)
│       │   └── 懒加载父文档索引 O(1) 反查
│       │
│       ├── index_construction.py — 索引构建
│       │   ├── OpenAIEmbeddings(百炼兼容)
│       │   │   ├── check_embedding_ctx_length=False ← 跳过 token 编码
│       │   │   └── chunk_size=10 ← 百炼单批上限
│       │   ├── FAISS.from_texts 向量化
│       │   ├── save_index / load_index 持久化
│       │   └── allow_dangerous_deserialization
│       │
│       ├── retrieval_optimization.py — 检索优化
│       │   ├── 向量检索 (k=5)
│       │   ├── BM25 检索 (k=5)
│       │   ├── RRF 重排: 1/(k+rank) 融合两路
│       │   │   └── 以 chunk_id 为融合键
│       │   └── metadata_filtered_search 元数据过滤
│       │
│       └── generation_integration.py — 生成集成
│           ├── ChatOpenAI(qwen-plus)
│           ├── 查询路由: list/detail/general
│           ├── 智能查询重写
│           ├── LLM 链: LCEL + StrOutputParser
│           ├── Prompt 模板集中管理
│           │   ├── BASIC_ANSWER / STEP_BY_STEP
│           │   └── QUERY_REWRITE / QUERY_ROUTER
│           └── 同步 _invoke / 流式 _stream
│
├── 🔄 问答主流程
│   └── 用户问题
│       ├── ① 路由分类 (list/detail/general)
│       ├── ② 查询重写 (list 保持原样)
│       ├── ③ 过滤条件提取 (分类/难度)
│       ├── ④ 混合检索 + RRF 重排 → top_k
│       ├── ⑤ 父文档反查 (去重排序)
│       └── ⑥ 按类型生成回答
│           ├── list → 菜名列表(不调 LLM)
│           ├── detail → 分步指引(菜品介绍/食材/步骤/技巧)
│           └── general → 基础回答
│
├── 🛠️ 已优化/修复
│   ├── 嵌入 API 400: 百炼只收字符串 → 跳过 tiktoken 编码
│   ├── 批量上限 10: chunk_size=10 分批
│   ├── 重载 ID 不一致: 确定性哈希 + resolve() 规范化
│   ├── 索引兼容性校验: chunk_id 抽样比对, 不符自动重建
│   ├── 难度过滤: "简单" 自动包含 "非常简单"
│   ├── 流式+list 兼容: isinstance 分支
│   ├── 绝对路径: 不依赖 CWD
│   ├── 模板/半成品: 排除 template, 新增半成品分类
│   └── 死代码清理: parent_child_map/batch_index 语义
│
└── ⚠️ 遗留问题
    ├── faiss AVX2 告警: 非 AVX2 CPU 回退, 仅噪音
    ├── 检索噪声: 偶现无关块(BM25 词重叠)
    └── pyproject 项目名 demo2 ≠ 目录 demo3
```

---

# 链路思维导图（处理链）

## 🔗 启动与构建链

```
python main.py
│
└─ main()
   └─ RecipeRAGSystem(config)
      ├─ ① 校验: data_path 存在 + LLM_API_KEY 已设置
      │
      ├─ ② initialize_system()
      │  ├─ DataPreparationModule(data_path)
      │  ├─ IndexConstructionModule(embedding_model, index_path)
      │  │  └─ setup_embeddings → OpenAIEmbeddings(百炼兼容)
      │  └─ GenerationIntegrationModule(llm_model)
      │     └─ _setup_llm → ChatOpenAI(qwen-plus)
      │
      └─ ③ build_knowledge_base()
         ├─ load_documents()  ──→ 322 篇 + 元数据增强(分类/菜名/难度)
         ├─ chunk_documents() ──→ Markdown 标题分块 → 1758 块
         ├─ load_index() ── 持久化索引存在?
         │  ├─ 是 → _index_matches_chunks() 兼容性校验
         │  │       ├─ 一致 → 直接复用
         │  │       └─ 不一致 → 置空重建
         │  └─ 否 → build_vector_store() → FAISS.from_texts
         │          └─ save_index() → vector_index/
         └─ RetrievalOptimizationModule(vectorstore, chunks)
            ├─ 向量检索器 (k=5)
            └─ BM25Retriever (k=5)
```

## 🔗 问答链（核心链路）

```
ask_question(question, stream)
│
├─ ① query_router() ──→ LLM 分类
│      list / detail / general
│
├─ ② query_rewrite() ──→ 模糊查询改写
│      (list 类型保持原样, 跳过改写)
│
├─ ③ _extract_filters() ──→ 提取 分类/难度 过滤条件
│      例: "简单的素菜" → {"category":"素菜", "difficulty":["简单","非常简单"]}
│
├─ ④ 检索 (二选一)
│  ├─ 有过滤 → metadata_filtered_search()  纯向量 + 元数据过滤
│  └─ 无过滤 → hybrid_search()  混合检索
│      ├─ 向量检索 invoke ──┐
│      ├─ BM25 检索 invoke ──┼─→ RRF 重排 (1/(k+rank), chunk_id 融合)
│      └─ 取 top_k = 3 ──────┘
│
├─ ⑤ get_parent_documents() ──→ 父文档反查
│      懒索引 O(1) + 按命中块数降序去重
│
└─ ⑥ 生成 (按路由类型分流)
   ├─ list    → generate_list_answer()     不调 LLM, 直接列菜名
   ├─ detail  → generate_step_by_step_answer() 分步指导(介绍/食材/步骤/技巧)
   └─ general → generate_basic_answer()    基础回答
   └─ 输出: stream=True → 生成器 | False → str
```

## 🔗 数据链路

```
data/C8/cook/*.md (322 篇)
└─ load_documents()  rglob 扫描 + 跳过 template/
   └─ 元数据增强: 分类(路径) / 菜名(文件名) / 难度(★ 数)
      └─ Markdown 标题分块 (#/##/###)
         ├─ 确定性 ID: parent_id=md5(路径) / chunk_id=md5(父ID:序号)
         ├─ 向量化: text-embedding-v3 (批 10, 跳过 token 编码)
         ├─ FAISS 索引: vector_index/ (持久化)
         └─ BM25 关键词索引 (内存)
```

## 🔗 模块归属（链路 → 模块）

| 链路环节 | 所属模块 |
|---|---|
| 文档加载 / 元数据 / 分块 / 父文档反查 | `rag_modules/data_preparation.py` |
| 向量化 / FAISS 构建 / 持久化 | `rag_modules/index_construction.py` |
| 混合检索 / RRF 重排 / 元数据过滤 | `rag_modules/retrieval_optimization.py` |
| 路由 / 重写 / 回答生成 / Prompt | `rag_modules/generation_integration.py` |
| 编排以上 4 模块 | `main.py` |