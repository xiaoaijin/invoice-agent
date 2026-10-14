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


class CompanyConfig(BaseModel):
    name: str
    tax_id: str


class PathsConfig(BaseModel):
    db: str = "data/invoice_agent.db"
    chroma_dir: str = "data/chroma"
    knowledge_dir: str = "knowledge"


class AppConfig(BaseModel):
    company: CompanyConfig
    risk: RiskConfig
    paths: PathsConfig

    def resolve(self, rel: str) -> Path:
        """把 config.yaml 中的相对路径解析为相对项目根的绝对路径。"""
        p = Path(rel)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def db_path(self) -> Path:
        return self.resolve(self.paths.db)

    @property
    def chroma_dir(self) -> Path:
        return self.resolve(self.paths.chroma_dir)

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
        raise RuntimeError(
            "DASHSCOPE_API_KEY 未配置。请复制 .env.example 为 .env 并填入真实 key。"
        )
    return key
