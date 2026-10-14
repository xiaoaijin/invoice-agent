"""配置加载：.env（密钥，pydantic-settings）+ config.yaml（业务配置）。

启动时 fail-fast：必填项缺失立即报错并给出明确提示。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """来自 .env 的密钥/模型配置。"""

    dashscope_api_key: str = ""
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model_vision: str = "qwen-vl-max"
    model_chat: str = "qwen-max"
    embedding_model: str = "text-embedding-v3"
    verify_provider: str = "mock"
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


class FeedbackConfig(BaseModel):
    """自反馈优化配置（各环路独立开关，LLM 失败自动降级）。"""

    answer: AnswerFeedbackConfig = Field(default_factory=AnswerFeedbackConfig)
    field_check: FieldCheckConfig = Field(default_factory=FieldCheckConfig)
    risk_review: RiskReviewConfig = Field(default_factory=RiskReviewConfig)


class MemoryConfig(BaseModel):
    """pgvector 长期记忆召回参数。"""

    recall_top_k: int = 5
    enabled: bool = True  # PG 不可用时自动降级，此开关用于彻底关闭


class CompanyConfig(BaseModel):
    name: str
    tax_id: str


class PathsConfig(BaseModel):
    db: str = "data/invoice_agent.db"
    knowledge_dir: str = "knowledge"


class AppConfig(BaseModel):
    company: CompanyConfig
    risk: RiskConfig
    context: ContextConfig = ContextConfig()
    memory: MemoryConfig = MemoryConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    feedback: FeedbackConfig = FeedbackConfig()
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
