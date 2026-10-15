"""prompt 版本管理：YAML 文件为源（prompts/ 目录），注册表按名解析当前版本。

设计（见 plans / config.yaml [prompts]）：
- 每个 prompt 一个 YAML：name / description / active / versions[{version, weight, description, text}]
- 解析优先级：显式 version 参数 > config.yaml prompts.overrides > A/B 加权抽样（sticky_key
  非 None 时改为确定性分桶）> YAML 内 active
- use() 在解析的同时设置 contextvar，monitoring handler 于 on_llm_end/on_llm_error 读取，
  写入 llm_calls 的 prompt_name/prompt_version/prompt_hash 三列
- 加载即把全部版本快照 upsert 到 monitor.db（旁路，失败不阻塞）
- prompt 缺失/损坏 fail-fast（同 config.py 哲学）；快照写失败静默降级（同 monitoring 哲学）
"""

from __future__ import annotations

import hashlib
import logging
import random
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError

from invoice_agent.config import get_app_config

logger = logging.getLogger(__name__)


class PromptNotFoundError(KeyError):
    """prompt 名不存在（fail-fast）。"""


class PromptFormatError(ValueError):
    """YAML 结构不合法：name 与文件名不一致 / active 指向不存在版本 / 版本号重复等。"""


@dataclass(frozen=True)
class PromptMeta:
    """一次解析结果：调用方用它组消息，监控侧据此落库。"""

    name: str
    version: str
    text: str
    hash: str  # sha256(text) 前 12 位，用于跨库追溯同一文本


class PromptVersion(BaseModel):
    version: str
    text: str
    weight: int = 1  # A/B 权重（仅 ab_enabled 时参与抽样），≥1
    description: str = ""


class PromptFile(BaseModel):
    name: str
    description: str = ""
    active: str
    versions: list[PromptVersion] = Field(min_length=1)

    def by_version(self, version: str) -> PromptVersion | None:
        return next((v for v in self.versions if v.version == version), None)

    def validate_active(self) -> None:
        if self.by_version(self.active) is None:
            raise PromptFormatError(f"prompt {self.name!r} 的 active={self.active!r} 不在 versions 中")


# 当前 LLM 调用使用的 prompt（use() set/reset，handler 读）
_CURRENT_PROMPT: ContextVar[PromptMeta | None] = ContextVar("current_prompt", default=None)


def current_prompt() -> PromptMeta | None:
    """handler 在 on_llm_end/on_llm_error 中读取；无上下文时返回 None（三列写 NULL）。"""
    return _CURRENT_PROMPT.get()


@dataclass
class _Rng:
    """实例内随机源，便于测试注入。"""

    inner: random.Random = field(default_factory=random.Random)

    def pick(self, items: list[PromptVersion]) -> PromptVersion:
        total = sum(max(1, v.weight) for v in items)
        point = self.inner.uniform(0, total)
        acc = 0.0
        for v in items:
            acc += max(1, v.weight)
            if point <= acc:
                return v
        return items[-1]

    def pick_sticky(self, items: list[PromptVersion], name: str, key: str) -> PromptVersion:
        total = sum(max(1, v.weight) for v in items)
        digest = hashlib.sha256(f"{name}:{key}".encode()).hexdigest()
        point = int(digest, 16) % total
        acc = 0
        for v in items:
            acc += max(1, v.weight)
            if point < acc:
                return v
        return items[-1]


class PromptRegistry:
    """加载 prompts/*.yaml 并解析版本；load 时把快照落库。"""

    def __init__(
        self,
        prompts_dir: Path,
        *,
        snapshot: bool = True,
        ab_enabled: bool = False,
        overrides: dict[str, str] | None = None,
    ) -> None:
        self.dir = prompts_dir
        self.ab_enabled = ab_enabled
        self.overrides = dict(overrides or {})
        self._rng = _Rng()
        self._files = self._load_all(prompts_dir)
        if snapshot:
            self._snapshot_all()

    # ---- 加载 ----

    def _load_all(self, prompts_dir: Path) -> dict[str, PromptFile]:
        if not prompts_dir.is_dir():
            raise FileNotFoundError(f"prompt 目录不存在：{prompts_dir}")
        files: dict[str, PromptFile] = {}
        for path in sorted(prompts_dir.glob("*.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            try:
                pf = PromptFile.model_validate(raw)
            except ValidationError as exc:
                raise PromptFormatError(f"prompt 文件不合法：{path}（{exc}）") from exc
            if pf.name != path.stem:
                raise PromptFormatError(f"prompt name={pf.name!r} 与文件名 {path.stem!r} 不一致")
            pf.validate_active()
            names = [v.version for v in pf.versions]
            if len(names) != len(set(names)):
                raise PromptFormatError(f"prompt {pf.name!r} 存在重复版本号")
            files[pf.name] = pf
        if not files:
            raise PromptFormatError(f"prompt 目录为空：{prompts_dir}")
        return files

    # ---- 解析 ----

    def get(
        self, name: str, *, version: str | None = None, sticky_key: str | None = None
    ) -> PromptMeta:
        """解析出当前生效版本。调用方禁止在 use() 块内再次 get 同名 prompt。"""
        pf = self._files.get(name)
        if pf is None:
            raise PromptNotFoundError(name)

        chosen: PromptVersion | None = None
        if version is not None:  # 显式指定，最高优先级
            chosen = pf.by_version(version)
            if chosen is None:
                raise PromptFormatError(f"prompt {name!r} 无版本 {version!r}")
        if chosen is None and name in self.overrides:
            ov = self.overrides[name]
            chosen = pf.by_version(ov)
            if chosen is None:
                raise PromptFormatError(f"overrides[{name!r}]={ov!r} 不是有效版本")
        if chosen is None and self.ab_enabled:
            variants = [v for v in pf.versions if max(1, v.weight) > 0]
            if len(variants) > 1:
                chosen = (
                    self._rng.pick_sticky(variants, name, sticky_key)
                    if sticky_key
                    else self._rng.pick(variants)
                )
        if chosen is None:
            chosen = pf.by_version(pf.active)
            assert chosen is not None  # 加载时已校验 active 有效

        return PromptMeta(
            name=name,
            version=chosen.version,
            text=chosen.text,
            hash=hashlib.sha256(chosen.text.encode()).hexdigest()[:12],
        )

    # ---- 查询 / 管理 ----

    def list_prompts(self) -> list[dict]:
        rows = []
        for name, pf in self._files.items():
            try:
                active_hash = hashlib.sha256(
                    pf.by_version(pf.active).text.encode()  # type: ignore[union-attr]
                ).hexdigest()[:12]
            except AttributeError:  # pragma: no cover — 加载时已校验，防御
                active_hash = ""
            rows.append(
                {
                    "name": name,
                    "description": pf.description,
                    "active": pf.active,
                    "active_hash": active_hash,
                    "versions": len(pf.versions),
                    "weights": {v.version: max(1, v.weight) for v in pf.versions},
                }
            )
        return rows

    def versions(self, name: str) -> list[PromptVersion]:
        pf = self._files.get(name)
        if pf is None:
            raise PromptNotFoundError(name)
        return list(pf.versions)

    def set_active(self, name: str, version: str) -> None:
        """切换激活版本：改内存并只替换 YAML 中 `active:` 一行（保留文件内注释与格式）。"""
        pf = self._files.get(name)
        if pf is None:
            raise PromptNotFoundError(name)
        if pf.by_version(version) is None:
            raise PromptFormatError(f"prompt {name!r} 无版本 {version!r}")
        pf.active = version
        path = self.dir / f"{name}.yaml"
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        replaced = False
        for i, line in enumerate(lines):
            if re.match(r"^active\s*:", line):
                lines[i] = f"active: {version}\n"
                replaced = True
                break
        if not replaced:
            raise PromptFormatError(f"prompt 文件缺少 active 字段：{path}")
        path.write_text("".join(lines), encoding="utf-8")
        logger.info("prompt %s 激活版本切换为 %s（已回写 %s）", name, version, path)

    # ---- 快照 ----

    def _snapshot_all(self) -> None:
        """把全部版本快照 upsert 到 monitor.db；失败静默（旁路不反噬主流程）。"""
        try:
            from invoice_agent.monitoring.store import MonitorStore

            store = MonitorStore(get_app_config().monitoring_db_path)
            for name, pf in self._files.items():
                for v in pf.versions:
                    store.upsert_prompt_version(
                        name=name,
                        version=v.version,
                        hash=hashlib.sha256(v.text.encode()).hexdigest()[:12],
                        description=v.description,
                        text=v.text,
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("prompt 快照落库失败（跳过）：%s", exc)


@lru_cache
def get_registry() -> PromptRegistry:
    """全局单例；路径与开关来自 get_app_config()。测试用 cache_clear() 隔离。"""
    cfg = get_app_config()
    return PromptRegistry(
        cfg.prompts_dir,
        snapshot=cfg.monitoring.enabled,
        ab_enabled=cfg.prompts.ab_enabled,
        overrides=cfg.prompts.overrides,
    )


@contextmanager
def use(
    name: str, *, version: str | None = None, sticky_key: str | None = None
) -> Iterator[PromptMeta]:
    """解析 prompt 并登记到 contextvar；with 块必须包住 LLM invoke，块内禁止再 get 同名 prompt。"""
    meta = get_registry().get(name, version=version, sticky_key=sticky_key)
    token = _CURRENT_PROMPT.set(meta)
    try:
        yield meta
    finally:
        _CURRENT_PROMPT.reset(token)
