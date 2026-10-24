"""配置加载：.env（密钥，pydantic-settings）+ config.yaml（业务配置）。

启动时 fail-fast：必填项缺失立即报错并给出明确提示。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from invoice_agent.models.invoice import ExpenseCategory

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """来自 .env 的密钥/模型配置。"""

    dashscope_api_key: str = ""
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model_vision: str = "qwen3.5-omni-plus"
    model_chat: str = "qwen3.8-max"
    embedding_model: str = "qwen3.7-text-embedding"
    verify_provider: str = "mock"
    # 报销审批流程提供方：mock（默认，本地打桩）| feishu（需配置下方凭证）
    expense_flow_provider: str = "mock"
    feishu_app_id: str = ""
    feishu_app_secret: str = ""
    feishu_approval_code: str = ""
    pg_dsn: str = "postgresql://invoice:invoice@localhost:5432/invoice_agent"

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


class RiskConfig(BaseModel):
    serial_gap_tolerance: int = 5
    split_amount_threshold: float = 5000.0
    split_window_days: int = 7
    split_min_count: int = 3
    reimburse_deadline_days: int = 90
    confidence_threshold: float = 0.8
    sensitive_keywords: list[str] = Field(default_factory=list)


class ContextConfig(BaseModel):
    """单次 LLM 交互的上下文预算与压缩策略。"""

    token_budget: int = 8000  # 单次交互 token 上限（字符估算）
    compress_threshold: float = 0.6  # 历史达到预算的 60% 时触发摘要压缩
    keep_recent_rounds: int = 10  # 压缩时原样保留的最近轮数（更早的进摘要）


class AlertConfig(BaseModel):
    """监控告警阈值。"""

    node_latency_budget_ms: dict[str, int] = Field(
        default_factory=lambda: {"default": 10000}
    )  # 节点延迟预算（按节点名查表，缺省用 default）
    run_latency_budget_ms: int = 60000
    max_retries_per_run: int = 2  # 单次 run 内同节点重复执行超此数告警
    success_rate_window: int = 20  # 近 N 次 run 的成功率窗口
    min_success_rate: float = 0.9


class MonitoringConfig(BaseModel):
    """全流程监控配置。"""

    enabled: bool = True
    db: str = "data/monitor.db"
    log_level: str = "INFO"
    log_file: str = "logs/agent.log"
    pricing: dict[str, dict[str, float]] = Field(default_factory=dict)  # 每千 token 成本（元）
    alert: AlertConfig = Field(default_factory=AlertConfig)


class AnswerFeedbackConfig(BaseModel):
    enabled: bool = True
    max_retries: int = 1  # 不合格时最多重生成次数
    pass_score: float = 0.7


class FieldCheckConfig(BaseModel):
    enabled: bool = True


class RiskReviewConfig(BaseModel):
    enabled: bool = True


class LessonsConfig(BaseModel):
    """自评教训持久化与召回（pgvector，PG 不可用自动降级）。"""

    enabled: bool = True
    recall_top_k: int = 3
    min_score: float = 0.6  # 余弦相似度阈值（text-embedding-v3 相关问题约 0.55+）


class IntentConfig(BaseModel):
    """意图识别漏斗（关键词 → 正则 → LLM 兜底）。"""

    enabled: bool = True  # 关闭时所有问题按 new_question 处理（不调 LLM）
    llm_fallback: bool = True  # 关键词/正则均未定性时是否允许 LLM 兜底
    chat_keywords: list[str] = Field(
        default_factory=lambda: ["你好", "在吗", "谢谢", "辛苦", "再见", "你是谁", "嗨"]
    )  # 寒暄/闲聊关键词（子串匹配）
    domain_keywords: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "policy": ["政策", "规定", "文件", "税法", "标准"],
            "header": ["抬头", "税号", "开票信息", "开票名称"],
            "deduction": ["抵扣", "进项", "税前扣除", "专票抵扣"],
            "reimburse": ["报销", "贴票", "报销流程", "报销时效"],
            "verify": ["查验", "验真", "真伪", "查验平台"],
        }
    )  # 业务领域关键词：命中最多的领域胜出，零成本打标


class FeedbackConfig(BaseModel):
    """自反馈优化配置（各环路独立开关，LLM 失败自动降级）。"""

    answer: AnswerFeedbackConfig = Field(default_factory=AnswerFeedbackConfig)
    field_check: FieldCheckConfig = Field(default_factory=FieldCheckConfig)
    risk_review: RiskReviewConfig = Field(default_factory=RiskReviewConfig)
    lessons: LessonsConfig = Field(default_factory=LessonsConfig)


class MemoryConfig(BaseModel):
    """pgvector 长期记忆召回参数。"""

    recall_top_k: int = 5
    enabled: bool = True  # PG 不可用时自动降级，此开关用于彻底关闭


class CheckpointerConfig(BaseModel):
    """LangGraph checkpointer 持久化后端。

    memory：进程内（默认，单元测试/单进程演示）；postgres：跨进程持久化（多轮会话
    历史落 PG，重启/多副本可恢复）。backend=postgres 且 PG 不可用时启动 fail-fast
    （不走静默降级——降级到 memory 会导致多 worker/多副本间静默串话）。
    """

    backend: Literal["memory", "postgres"] = "memory"
    dsn: str | None = None  # 留空复用 .env 的 PG_DSN


class CompanyConfig(BaseModel):
    name: str
    tax_id: str


class PromptsConfig(BaseModel):
    """prompt 版本管理：YAML 文件为源（prompts/ 目录），此处只放运行时开关。"""

    dir: str = "prompts"
    ab_enabled: bool = False  # A/B 抽样总开关（按 versions[].weight 加权）
    overrides: dict[str, str] = Field(default_factory=dict)  # 强制版本，优先级最高，如 qa: v2


class ReimburseConfig(BaseModel):
    """批量报销批次配置。"""

    title_prefix: str = "员工费用报销"  # 批次标题前缀，UI 可覆盖
    categories: list[str] = Field(
        default_factory=lambda: [c.value for c in ExpenseCategory]
    )  # 与 ExpenseCategory 枚举值保持一致（tests/test_expense_models.py 把关）


class PathsConfig(BaseModel):
    db: str = "data/invoice_agent.db"
    knowledge_dir: str = "knowledge"


class UploadConfig(BaseModel):
    """上传文件校验：大小/类型在 UI 与 API 入口检查，页数上限在 intake 节点兜底。"""

    max_size_mb: int = 10
    max_pdf_pages: int = 5  # 超出页数只取前 N 页（防大 PDF 全页转 base64 爆内存）
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [".png", ".jpg", ".jpeg", ".pdf"]
    )
    image_max_long_edge_px: int = 2200  # 识票前长边缩放上限（vision 输入 token 随像素数涨）
    image_jpeg_quality: int = 85  # 识票前统一 JPEG 有损重编码质量（原 PNG 强转体积更大）
    pdf_render_dpi: int = 150  # PDF 逐页渲染 DPI（intake 转图用）


class AppConfig(BaseModel):
    company: CompanyConfig
    risk: RiskConfig
    context: ContextConfig = ContextConfig()
    memory: MemoryConfig = MemoryConfig()
    checkpointer: CheckpointerConfig = Field(default_factory=CheckpointerConfig)
    monitoring: MonitoringConfig = MonitoringConfig()
    intent: IntentConfig = Field(default_factory=IntentConfig)
    feedback: FeedbackConfig = FeedbackConfig()
    prompts: PromptsConfig = PromptsConfig()
    upload: UploadConfig = Field(default_factory=UploadConfig)
    reimburse: ReimburseConfig = Field(default_factory=ReimburseConfig)
    paths: PathsConfig

    def resolve(self, rel: str) -> Path:
        """把 config.yaml 中的相对路径解析为相对项目根的绝对路径。"""
        p = Path(rel)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def db_path(self) -> Path:
        return self.resolve(self.paths.db)

    @property
    def monitoring_db_path(self) -> Path:
        return self.resolve(self.monitoring.db)

    @property
    def knowledge_dir(self) -> Path:
        return self.resolve(self.paths.knowledge_dir)

    @property
    def prompts_dir(self) -> Path:
        return self.resolve(self.prompts.dir)


def load_app_config(path: Path | None = None) -> AppConfig:
    cfg_path = path or PROJECT_ROOT / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"业务配置文件不存在：{cfg_path}")
    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig.model_validate(raw)


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_app_config() -> AppConfig:
    return load_app_config()


def require_api_key() -> str:
    """需要真实调用 LLM 时使用；缺失则 fail-fast 并给出可操作的提示。"""
    key = get_settings().dashscope_api_key
    if not key or key.startswith("sk-xxxx"):
        raise RuntimeError("DASHSCOPE_API_KEY 未配置。请复制 .env.example 为 .env 并填入真实 key。")
    return key
