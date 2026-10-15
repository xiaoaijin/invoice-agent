# 发票助手 Agent（MVP + 知识增强风控）

基于 **LangGraph + Qwen 系列（阿里云 MaaS 专属端点 / DashScope 兼容模式）** 的发票助手：智能识票、验真、重复报销检测、风控预警、带引用的政策问答。模型经 `.env` 配置：识票 `MODEL_VISION`（默认 qwen3.5-omni-plus）、问答 `MODEL_CHAT`（默认 qwen3.8-max）、嵌入 `EMBEDDING_MODEL`（默认 qwen3.7-text-embedding）。

## 功能

| 功能 | 说明 |
|---|---|
| 智能识票 | 图片/PDF → 结构化 JSON（发票类型/代码/号码/日期/购销双方/金额/明细），含字段级置信度 |
| 合规查验 | 验真（Mock Provider，接口抽象可切换百望云/诺诺）、抬头税号校验、重复报销检测（SQLite 台账） |
| 政策问答 | RAG 混合检索（pgvector 语义 + BM25/jieba，RRF 融合），回答强制引用条款编号，检索不足时明确拒答 |
| 异常预警 | 8 条规则：抬头不符 / 重复 / 连号 / 大额拆分 / 敏感商品 / 报销时效 / 日期异常 / 金额勾稽 |
| 多轮对话 | checkpointer（MemorySaver）按 thread_id 持久化会话；支持追问（检索 query 自动拼上上一问题）；发票查验后可在对话中继续追问 |
| 上下文管理 | 单次交互 token 上限（字符估算）；历史达预算 60% 时自动压缩：最近 10 轮原样保留，更早内容滚动摘要化，工具调用只留工具名+结果；prompt 组装后还有预算兜底截断 |
| 长期记忆 | 每轮问答以向量存入 PostgreSQL (pgvector)，跨会话语义召回相关历史；会话滚动摘要持久化，重启不丢；PG 不可用时自动降级 |
| Prompt 版本管理 | `prompts/*.yaml` 为唯一源（多版本 + 激活版本 + A/B 权重），解析优先级：显式指定 > config overrides > A/B 加权抽样 > YAML `active`；每次 LLM 调用把 prompt 名/版本/内容 hash 写入监控库，支持调用链追溯与按版本统计调用量/成功率/成本 |
| 全流程监控 | 独立 SQLite（`data/monitor.db`）记录 run/node span、延迟、token 用量与成本（按 config.yaml 价目折算），超预算/低成功率告警；旁路设计，写失败不反噬主流程 |
| 自反馈优化 | 回答质量 LLM 自评（不合格重生成 ≤1 次）、识票字段 LLM 级自检、风控结论 LLM 复核（误报移除/漏报补充）；LLM 不可用时静默降级放行 |
| 界面 | Streamlit：千问式多会话聊天（会话列表、标题取首问、侧边栏可收起）+ 流式输出 + 上传查验 + 确认报销入台账；会话历史持久化到 SQLite（`data/chat_history.db`），重启可恢复 |

## 快速开始

```bash
# 1. 安装 uv（已装可跳过）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 安装依赖（自动下载 Python 3.12）
uv sync

# 3. 启动 PostgreSQL (pgvector)——知识库 RAG 与长期记忆存储
docker compose up -d
# 无 Docker 环境可用 scripts/local_pg.py（pgserver 自包含二进制，免装 Docker/编译）：
uv pip install --default-index https://pypi.tuna.tsinghua.edu.cn/simple pgserver
uv run --no-sync python scripts/local_pg.py

# 4. 配置 API Key
cp .env.example .env   # 编辑 .env，填入 DASHSCOPE_API_KEY（PG_DSN 默认连本地 docker 实例）

# 5. 生成合成发票样本（可选，用于演示/评测）
uv run python scripts/make_sample_invoices.py

# 6. 预置 Mock 数据（假票/真票注册表 + 历史报销记录）
uv run python scripts/seed_mock_data.py

# 7. 构建 RAG 知识库索引（写入 pgvector）
uv run python scripts/build_knowledge_index.py

# 8. 启动界面
uv run streamlit run src/invoice_agent/ui/app.py
```

> 没有 Docker/PostgreSQL？应用仍可运行：RAG 与长期记忆自动降级（问答无引用、无跨会话记忆），其余功能不受影响。
>
> `scripts/local_pg.py` 说明：数据目录在项目内 `.pgdata/`（已 gitignore）；幂等可重复执行，进程退出后 PG 保持常驻；停止用 `pg_ctl -D .pgdata stop`。机器无法直连 pypi.org 时，安装 pgserver 需走清华镜像（见上方命令）。

CLI 调试：

```bash
uv run python -m invoice_agent.cli recognize samples/01_normal_special.png  # 只识票，输出 JSON
uv run python -m invoice_agent.cli run samples/04_wrong_taxid.png           # 完整链路
uv run python -m invoice_agent.cli ask "餐饮发票能抵扣进项税吗"               # 纯政策问答
uv run python -m invoice_agent.cli chat                                      # 多轮对话 REPL（/invoice <文件> 查票，/new 新会话）
uv run python -m invoice_agent.cli prompts list                              # prompt 版本管理：列出全部 prompt 及激活版本
uv run python -m invoice_agent.cli prompts show qa --version v2              # 查看 prompt 全文
uv run python -m invoice_agent.cli prompts set-active qa v1                  # 切换激活版本（回写 YAML）
uv run python -m invoice_agent.cli prompts usage                             # 按 prompt 名+版本统计调用量/成功率/成本
uv run python -m invoice_agent.cli serve                                     # 启动 HTTP API（默认 127.0.0.1:8000）
```

## HTTP API（供 ERP/Java 等外部系统调用）

```bash
uv run invoice-agent serve --host 127.0.0.1 --port 8000   # FastAPI/uvicorn，交互文档见 /docs
```

| 端点 | 说明 |
|---|---|
| `GET /healthz` | 健康检查（config 可加载 + data 目录可写；不探 PG——PG 挂了主流程自动降级） |
| `POST /invoices` | multipart 上传发票（`file`，可选 `question`/`session_id`）→ 完整链路，返回识票/验真/查重/风控/回答 JSON |
| `POST /ask` | `{"question"}` 纯政策问答 |
| `POST /chat` | `{"message", "session_id"?}` 会话式多轮（同 session_id 共享记忆；缺省时服务端生成并回传） |
| `GET /chat/{session_id}/history` | 会话历史（与 UI 的 `data/chat_history.db` 互通） |
| `POST /chat/stream` | SSE 流式对话（`node`/`token`/`done` 三类事件） |
| `GET /runs/{run_id}` | 查监控 run 明细（节点时间线 + LLM 调用级 token/成本/prompt 版本） |

错误约定：LLM/PG 失败在节点内静默降级返回 200（业务结果含降级提示）；文件超大小上限 413、类型不符 415、run 不存在 404、服务未配置（如缺 API key）503。

> **部署约束**：LangGraph checkpointer 为进程内 `MemorySaver`，多轮会话记忆不跨进程/worker 共享，必须**单 worker** 部署（`serve` 已固定 `workers=1`）；跨重启历史恢复走 `GET /chat/{id}/history`。每个响应带的 `run_id` 可用于 `GET /runs/{id}` 追溯，也是后续异步任务扩展（202 + 轮询）的预留锚点。

## 演示路径（用合成样本）

| 样本 | 预期表现 |
|---|---|
| `01_normal_special.png` | 识别正确、验真 PASS、无风险 |
| `02_normal_personal.png` | 个人抬头无税号，正常通过 |
| `03_digital.png` | 数电票：20 位号码、无发票代码 |
| `04_wrong_taxid.png` | R001 高危：购买方税号与公司不符 |
| `05_sensitive_giftcard.png` | R005 高危：明细含"购物卡" |
| `06_serial_with_01.png` | R003：与台账中 28374651 连号 |
| `07_fake_invoice.png` | 验真 FAIL（Mock 注册表登记的假票） |
| `08_blurred.png` | 模糊图 → 提取失败 → 重试 → 降级人工复核 |

## 测试

```bash
uv run pytest                # 单元测试（无网络、无 LLM、无 PG）
uv run pytest --override-ini addopts="-m integration"   # 集成测试（需 API key + PostgreSQL）
```

## 架构

```
LangGraph 状态图：
START → intake → [route_entry] ─┬→ recognize ⇄（失败重试 ≤2）→ [check_extract]
                                │                                  ↓ 通过
                                │                      verify → dedup → risk
                                └→ retrieve_docs ←──────────────────────┘
                                        ↓
                                     finalize（对话模型汇总，含 [来源N] 引用）→ END
```

- **识票**：视觉模型 `MODEL_VISION` + `with_structured_output(InvoiceData)`，温度 0，缺失字段填 null 禁止编造；字段级 LLM 自检可复用提取重试边（`feedback.field_check`）
- **验真**：`InvoiceVerifyProvider` 抽象 + Mock 实现（注册表命中 → 确定性结果；未命中 → 哈希伪随机 85% PASS）。接入百望云/诺诺时新增 Provider 实现即可，图节点零改动
- **去重**：SQLite 台账，"确认提交报销"后才写入，避免查询污染连号/拆分判断；`file_sha256` 区分同文件重传
- **RAG**：按「第X条」切分保引用完整 → `EMBEDDING_MODEL`（qwen3.7-text-embedding）+ **PostgreSQL pgvector** 语义检索 → jieba+BM25 → RRF 融合 → 引用编号来自切分元数据（非模型生成）；qa prompt v2 去模板化——直接自然作答，保留 [来源N] 引用与不足即拒答约束
- **多轮记忆**：`MemorySaver` checkpointer 按 thread_id 持久化整个 AgentState（含 `chat_history`）；finalize 把历史转为消息注入 LLM；短问题/「那/为什么」开头的追问在检索时自动拼上上一问题补全语义；UI 多会话消息另存 SQLite `data/chat_history.db`（pydantic 模型带 `__type__` 标签序列化，重启恢复渲染）
- **上下文管理**（`services/context_manager.py`）：token 字符估算（CJK≈1/字、ASCII≈4字符/token）；历史超 `token_budget` 的 60% 触发压缩——最近 10 轮原样保留、更早内容由对话模型做滚动摘要（工具调用只留工具名+结果）、摘要存入 `history_summary` 与 PG `memory_summaries`；最终 prompt 由 `fit_budget` 兜底截断保证不超上限
- **长期记忆**（`services/memory_store.py`）：每轮问答 Q+A 合并嵌入存 PG `memory_chunks`（HNSW 索引），新问题向量召回 top-k 相关历史注入 prompt；PG 不可用时全部静默降级
- **Prompt 版本管理**（`prompts/registry.py` + `prompts/*.yaml`）：每个 prompt 一个 YAML（`versions`/`active`/`weight`）；解析优先级为显式 version 参数 > `config.yaml prompts.overrides` > A/B 加权抽样（`ab_enabled: true` 时按 weight 加权）> YAML 内 `active`。调用时经 contextvar 把 prompt 名/版本/hash（sha256 前 12 位）写入监控库 `llm_calls` 三列，全部版本快照旁路 upsert 到 monitor.db 供跨库追溯；YAML 缺失/损坏 fail-fast，快照写失败静默降级
- **全流程监控**（`monitoring/`）：runs / node_spans / llm_calls 三张表记 run 状态、节点延迟与重试、token 用量与成本（按 `config.yaml monitoring.pricing` 每千 token 折算），节点/run 延迟超预算、单 run 重试过多、近 20 次成功率 < 90% 触发告警；全部写入吞异常只打 warning，绝不反噬主流程
- **自反馈优化**（`feedback/critics.py`）：三个可注入 LLM 的 critic——回答质量自评（综合评分 < `pass_score` 时重生成，最多 `max_retries` 次）、识票字段自检、风控结论复核（误报移除/漏报补充，单次修订不回环）；LLM 为 None 或调用异常时静默降级放行
- **配置**：密钥在 `.env`（不入库），业务阈值/敏感词/上下文预算/监控告警/prompt 版本策略在 `config.yaml`

## 已知限制与注意事项

1. **验真为 Mock**：结果显式标注，不代表真实真伪。接入真实平台需企业资质与付费开通。
2. **知识库为合成语料**：`knowledge/` 按三层组织——`national/`（国家法规）、`local/`（地方细则，上海示例）、`company/`（公司制度），均为节选/示例文档（部分条款已过时如旅客运输抵扣政策），生产使用前需替换为最新法规全文、所在地区真实细则与公司真实制度，并重建索引。
3. **发票数据会发送至 DashScope API**（识票/嵌入/汇总）；业务数据存储在本地 SQLite 与自建 PostgreSQL。请勿上传真实敏感发票用于演示。
4. **token 上限为字符估算**（无 Qwen 本地 tokenizer），按保守口径设定（CJK≈1 token/字）；`context.token_budget` 可在 config.yaml 调整。
5. **pgvector 链路需在有 PostgreSQL 的环境验收**：本仓库 CI 无 PG，`docker compose up -d` 后运行 `uv run pytest --override-ini addopts="-m integration"` 验证 RAG 检索与记忆召回。
6. `samples/` 中的 PNG 已 gitignore；**真实发票严禁提交仓库**。
7. **合成样本准确率偏高**：真实发票的盖章/折痕/拍照光斑会降低识别率，上线验收请用脱敏真实发票复测；必要时在 Phase 2 引入 PaddleOCR 交叉验证。

## Phase 2 路线（未实现）

- PaddleOCR 交叉验证、DashScope gte-rerank 精排
- 验真接入百望云/诺诺真实 API
- 企业微信/钉钉/飞书机器人接入、批量异步处理
- LangGraph checkpointer 跨进程持久化（当前 MemorySaver 为进程内；UI 会话消息已持久化到 `data/chat_history.db`，生产可进一步换 SqliteSaver/PostgresSaver）
- Prompt A/B 评估看板（当前 `prompts usage` 已按版本统计调用量/成功率/成本，缺人工标注与对比报告）
