"""PromptRegistry 单测：加载校验、版本解析优先级、A/B 抽样、set_active 回写、快照落库。"""

import hashlib
from pathlib import Path

import pytest
import yaml

from invoice_agent.prompts.registry import (
    PromptFormatError,
    PromptNotFoundError,
    PromptRegistry,
    current_prompt,
    use,
)


def _write_prompt(
    dir_path: Path,
    name: str,
    versions: list[dict],
    active: str = "v1",
    description: str = "",
) -> Path:
    payload = {
        "name": name,
        "description": description,
        "active": active,
        "versions": versions,
    }
    path = dir_path / f"{name}.yaml"
    path.write_text(yaml.dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def _mk_registry(prompts_dir: Path, *, ab_enabled: bool = False, overrides=None) -> PromptRegistry:
    return PromptRegistry(
        prompts_dir,
        snapshot=False,
        ab_enabled=ab_enabled,
        overrides=overrides,
    )


@pytest.fixture
def prompts_dir(tmp_path: Path) -> Path:
    d = tmp_path / "prompts"
    d.mkdir()
    _write_prompt(
        d,
        "qa",
        [
            {"version": "v1", "text": "第一版", "weight": 1},
            {"version": "v2", "text": "第二版", "weight": 3},
        ],
    )
    return d


# ---- 加载与 fail-fast ----


class TestLoad:
    def test_get_returns_meta_and_hash(self, prompts_dir: Path):
        reg = _mk_registry(prompts_dir)
        meta = reg.get("qa")
        assert meta.name == "qa"
        assert meta.version == "v1"
        assert meta.text == "第一版"
        assert meta.hash == hashlib.sha256("第一版".encode()).hexdigest()[:12]

    def test_missing_name_fails_fast(self, prompts_dir: Path):
        reg = _mk_registry(prompts_dir)
        with pytest.raises(PromptNotFoundError):
            reg.get("nope")

    def test_missing_dir_fails_fast(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            PromptRegistry(tmp_path / "nope", snapshot=False)

    def test_active_must_exist(self, tmp_path: Path):
        d = tmp_path / "prompts"
        d.mkdir()
        _write_prompt(d, "qa", [{"version": "v1", "text": "x"}], active="v9")
        with pytest.raises(PromptFormatError):
            _mk_registry(tmp_path)

    def test_name_must_match_filename(self, tmp_path: Path):
        d = tmp_path / "prompts"
        d.mkdir()
        _write_prompt(d, "qa", [{"version": "v1", "text": "x"}])
        (d / "qa.yaml").rename(d / "other.yaml")
        with pytest.raises(PromptFormatError):
            _mk_registry(tmp_path)

    def test_duplicate_version_rejected(self, tmp_path: Path):
        d = tmp_path / "prompts"
        d.mkdir()
        _write_prompt(d, "qa", [{"version": "v1", "text": "x"}, {"version": "v1", "text": "y"}])
        with pytest.raises(PromptFormatError):
            _mk_registry(tmp_path)

    def test_hash_sensitive_to_text(self, prompts_dir: Path):
        reg = _mk_registry(prompts_dir)
        h1 = reg.get("qa").hash
        _write_prompt(prompts_dir, "qa", [{"version": "v1", "text": "第一版。"}])
        reg2 = PromptRegistry(prompts_dir, snapshot=False)
        assert reg2.get("qa").hash != h1


# ---- 版本解析优先级 ----


class TestResolvePriority:
    def test_explicit_version_wins(self, prompts_dir: Path):
        reg = _mk_registry(prompts_dir, overrides={"qa": "v2"})
        assert reg.get("qa", version="v1").version == "v1"

    def test_override_beats_active(self, prompts_dir: Path):
        reg = _mk_registry(prompts_dir, overrides={"qa": "v2"})
        assert reg.get("qa").version == "v2"

    def test_fallback_to_active(self, prompts_dir: Path):
        reg = _mk_registry(prompts_dir)
        assert reg.get("qa").version == "v1"

    def test_invalid_override_fails_fast(self, prompts_dir: Path):
        reg = _mk_registry(prompts_dir, overrides={"qa": "vX"})
        with pytest.raises(PromptFormatError):
            reg.get("qa")

    def test_ab_disabled_uses_active(self, prompts_dir: Path):
        reg = _mk_registry(prompts_dir, ab_enabled=False)
        assert reg.get("qa").version == "v1"

    def test_ab_enabled_samples(self, prompts_dir: Path):
        reg = _mk_registry(prompts_dir, ab_enabled=True)
        versions = {reg.get("qa").version for _ in range(100)}
        assert versions == {"v1", "v2"}  # 抽样会命中两个变体


# ---- A/B 抽样 ----


class TestABSampling:
    def test_weight_ratio(self, prompts_dir: Path, monkeypatch):
        import random as random_mod

        reg = _mk_registry(prompts_dir, ab_enabled=True)
        # 固定随机源保证可重复
        rng = random_mod.Random(42)
        monkeypatch.setattr(reg._rng, "inner", rng)
        counts = {"v1": 0, "v2": 0}
        for _ in range(1000):
            counts[reg.get("qa").version] += 1
        # 权重 1:3 → 比例应在 [0.125, 0.375] 区间（1000 次下方差足够小）
        assert 0.125 < counts["v1"] / 1000 < 0.375

    def test_sticky_key_deterministic(self, prompts_dir: Path):
        reg = _mk_registry(prompts_dir, ab_enabled=True)
        picks = {reg.get("qa", sticky_key="thread-A").version for _ in range(1000)}
        assert len(picks) == 1  # 同一 sticky_key 永远同一变体

    def test_sticky_key_distributes(self, prompts_dir: Path):
        reg = _mk_registry(prompts_dir, ab_enabled=True)
        versions = {reg.get("qa", sticky_key=f"t{i}").version for i in range(50)}
        assert len(versions) == 2  # 不同 key 落到不同变体


# ---- set_active / 查询 ----


class TestManage:
    def test_set_active_persists(self, prompts_dir: Path):
        reg = _mk_registry(prompts_dir)
        reg.set_active("qa", "v2")
        # 重新加载读到新 active
        reg2 = PromptRegistry(prompts_dir, snapshot=False)
        assert reg2.get("qa").version == "v2"
        # 回写文件可解析
        raw = yaml.safe_load((prompts_dir / "qa.yaml").read_text(encoding="utf-8"))
        assert raw["active"] == "v2"

    def test_set_active_invalid_version(self, prompts_dir: Path):
        reg = _mk_registry(prompts_dir)
        with pytest.raises(PromptFormatError):
            reg.set_active("qa", "vX")

    def test_list_prompts(self, prompts_dir: Path):
        reg = _mk_registry(prompts_dir)
        rows = {r["name"]: r for r in reg.list_prompts()}
        assert rows["qa"]["active"] == "v1"
        assert rows["qa"]["versions"] == 2
        assert rows["qa"]["weights"] == {"v1": 1, "v2": 3}

    def test_versions(self, prompts_dir: Path):
        reg = _mk_registry(prompts_dir)
        assert [v.version for v in reg.versions("qa")] == ["v1", "v2"]


# ---- 快照 ----


class TestSnapshot:
    def test_snapshot_upsert_idempotent(self, tmp_path: Path):
        import invoice_agent.prompts.registry as registry_mod
        from invoice_agent.config import (
            AppConfig,
            CompanyConfig,
            MonitoringConfig,
            PathsConfig,
            RiskConfig,
        )

        d = tmp_path / "prompts"
        d.mkdir()
        _write_prompt(d, "qa", [{"version": "v1", "text": "x"}, {"version": "v2", "text": "y"}])

        fake_cfg = AppConfig(
            company=CompanyConfig(name="x", tax_id="y"),
            risk=RiskConfig(),
            monitoring=MonitoringConfig(db=str(tmp_path / "m.db")),
            paths=PathsConfig(db=str(tmp_path / "g.db"), knowledge_dir=str(tmp_path / "k")),
        )
        monkey_target = registry_mod
        orig = monkey_target.get_app_config
        monkey_target.get_app_config = lambda: fake_cfg
        try:
            PromptRegistry(d, snapshot=True)
            PromptRegistry(d, snapshot=True)  # 重复加载不重复（主键 upsert）
        finally:
            monkey_target.get_app_config = orig

        from invoice_agent.monitoring.store import MonitorStore

        store = MonitorStore(tmp_path / "m.db")
        rows = store.list_prompt_versions()
        assert [r["version"] for r in rows] == ["v1", "v2"]
        assert rows[0]["hash"] == hashlib.sha256(b"x").hexdigest()[:12]

    def test_snapshot_failure_is_silent(self, tmp_path: Path):
        """快照目标不可写时不抛异常（旁路）。"""
        import invoice_agent.prompts.registry as registry_mod

        d = tmp_path / "prompts"
        d.mkdir()
        _write_prompt(d, "qa", [{"version": "v1", "text": "x"}])
        reg = PromptRegistry(d, snapshot=False)
        real = registry_mod.get_app_config
        registry_mod.get_app_config = lambda: None  # type: ignore[assignment]
        try:
            reg._snapshot_all()  # get_app_config() 返回 None → AttributeError → 吞掉
        finally:
            registry_mod.get_app_config = real


# ---- use() contextvar ----


class TestUse:
    def test_use_sets_and_resets(self, prompts_dir: Path):
        _mk_registry(prompts_dir)
        assert current_prompt() is None
        with use("qa") as meta:
            assert current_prompt() is meta
            assert current_prompt().name == "qa"  # type: ignore[union-attr]
        assert current_prompt() is None

    def test_use_resets_on_exception(self, prompts_dir: Path):
        _mk_registry(prompts_dir)
        with pytest.raises(RuntimeError), use("qa"):
            raise RuntimeError("x")
        assert current_prompt() is None


class TestRealPromptsDir:
    def test_real_prompts_load(self):
        """真实 prompts 目录可加载：critique v2 / query_rewrite v2 / intent 均存在且 active 可解析。

        防 name 与文件名 stem 不一致 / active 指向缺失版本（registry fail-fast 全局生效）。
        """
        from invoice_agent.config import get_app_config

        registry = PromptRegistry(get_app_config().prompts_dir, snapshot=False)
        critique = registry.get("critique")
        assert critique.version == "v2"
        assert "retrieval_issues" in critique.text
        rewrite = registry.get("query_rewrite")
        assert rewrite.name == "query_rewrite"
        assert rewrite.version == "v2"
        intent_prompt = registry.get("intent")
        assert intent_prompt.name == "intent"
        assert "retrieval_type" in intent_prompt.text
