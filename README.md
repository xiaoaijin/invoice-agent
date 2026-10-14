# 发票助手 Agent（MVP + 知识增强风控）

基于 **LangGraph + Qwen-VL-Max（DashScope 兼容模式）** 的发票助手：智能识票、验真、重复报销检测、风控预警、带引用的政策问答。

## 功能

| 功能 | 说明 |
|---|---|
| 智能识票 | 图片/PDF → 结构化 JSON（发票类型/代码/号码/日期/购销双方/金额/明细），含字段级置信度 |
| 合规查验 | 验真（Mock Provider，接口抽象可切换百望云/诺诺）、抬头税号校验、重复报销检测（SQLite 台账） |
| 政策问答 | RAG 混合检索（Chroma 向量 + BM25/jieba，RRF 融合），回答强制引用条款编号，检索不足时明确拒答 |
| 异常预警 | 8 条规则：抬头不符 / 重复 / 连号 / 大额拆分 / 敏感商品 / 报销时效 / 日期异常 / 金额勾稽 |
| 多轮对话 | checkpointer（MemorySaver）按 thread_id 持久化会话；支持追问（检索 query 自动拼上上一问题）；发票查验后可在对话中继续追问 |
| 界面 | Streamlit：上传查验 + 确认报销入台账 + 多轮对话聊天（可一键新会话） |

## 快速开始

```bash
# 1. 安装 uv（已装可跳过）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 安装依赖（自动下载 Python 3.12）
uv sync

# 3. 配置 API Key
cp .env.example .env   # 编辑 .env，填入 DASHSCOPE_API_KEY

# 4. 生成合成发票样本（可选，用于演示/评测）
uv run python scripts/make_sample_invoices.py

# 5. 预置 Mock 数据（假票/真票注册表 + 历史报销记录）
uv run python scripts/seed_mock_data.py

# 6. 构建 RAG 知识库索引
uv run python scripts/build_knowledge_index.py

# 7. 启动界面
uv run streamlit run src/invoice_agent/ui/app.py
```

CLI 调试：

```bash
uv run python -m invoice_agent.cli recognize samples/01_normal_special.png  # 只识票，输出 JSON
uv run python -m invoice_agent.cli run samples/04_wrong_taxid.png           # 完整链路
uv run python -m invoice_agent.cli ask "餐饮发票能抵扣进项税吗"               # 纯政策问答
uv run python -m invoice_agent.cli chat                                      # 多轮对话 REPL（/invoice <文件> 查票，/new 新会话）
```

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
uv run pytest                # 单元测试（无网络、无 LLM）
uv run pytest --override-ini addopts="-m integration"   # 集成测试（需 API key）
```

## 架构

```
LangGraph 状态图：
START → intake → [route_entry] ─┬→ recognize ⇄（失败重试 ≤2）→ [check_extract]
                                │                                  ↓ 通过
                                │                      verify → dedup → risk
                                └→ retrieve_docs ←──────────────────────┘
                                        ↓
                                     finalize（qwen-max 汇总，含 [来源N] 引用）→ END
```

- **识票**：`qwen-vl-max` + `with_structured_output(InvoiceData)`，温度 0，缺失字段填 null 禁止编造
- **验真**：`InvoiceVerifyProvider` 抽象 + Mock 实现（注册表命中 → 确定性结果；未命中 → 哈希伪随机 85% PASS）。接入百望云/诺诺时新增 Provider 实现即可，图节点零改动
- **去重**：SQLite 台账，"确认提交报销"后才写入，避免查询污染连号/拆分判断；`file_sha256` 区分同文件重传
- **RAG**：按「第X条」切分保引用完整 → DashScope text-embedding-v3 + Chroma → jieba+BM25 → RRF 融合 → 引用编号来自切分元数据（非模型生成）
- **多轮记忆**：`MemorySaver` checkpointer 按 thread_id 持久化整个 AgentState（含 `chat_history`）；finalize 把最近 8 条历史转为消息注入 LLM；短问题/「那/为什么」开头的追问在检索时自动拼上上一问题补全语义
- **配置**：密钥在 `.env`（不入库），业务阈值/敏感词在 `config.yaml`

## 已知限制与注意事项

1. **验真为 Mock**：结果显式标注，不代表真实真伪。接入真实平台需企业资质与付费开通。
2. **知识库为合成语料**（`knowledge/` 下为节选/示例文档，部分条款已过时如旅客运输抵扣政策），生产使用前需替换为最新法规全文与公司真实制度，并重建索引。
3. **发票数据会发送至 DashScope API**（识票/嵌入/汇总）；数据存储全部本地。请勿上传真实敏感发票用于演示。
4. **合成样本准确率偏高**：真实发票的盖章/折痕/拍照光斑会降低识别率，上线验收请用脱敏真实发票复测；必要时在 Phase 2 引入 PaddleOCR 交叉验证。
5. `samples/` 中的 PNG 已 gitignore；**真实发票严禁提交仓库**。

## Phase 2 路线（未实现）

- PaddleOCR 交叉验证、DashScope gte-rerank 精排
- FastAPI 服务层（供 ERP/Java 系统调用）
- 验真接入百望云/诺诺真实 API
- 企业微信/钉钉/飞书机器人接入、批量异步处理
- 跨进程持久化记忆（当前 MemorySaver 为进程内，重启即失；生产可换 SqliteSaver/PostgresSaver）
