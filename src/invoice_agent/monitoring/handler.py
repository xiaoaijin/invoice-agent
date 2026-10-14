"""TraceCallbackHandler：LangChain 回调采集全流程 trace。

- root chain（parent_run_id 为 None）→ runs 表
- 已知节点 chain → node_spans 表（同节点重复执行计 retry_of）
- chat model 调用 → llm_calls 表（token 用量 + 成本，取不到记 null/0）
- run 结束时按阈值评估五类告警

所有回调方法体包 try/except：监控失败只打 debug 日志，不影响主流程。
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from invoice_agent.config import MonitoringConfig
from invoice_agent.monitoring.cost import calc_cost
from invoice_agent.monitoring.store import MonitorStore

logger = logging.getLogger(__name__)

# 已知图节点（chain run name 即节点名）
KNOWN_NODES = {
    "intake",
    "recognize",
    "verify",
    "dedup",
    "risk",
    "risk_review",
    "retrieve_docs",
    "finalize",
    "critic_answer",
}


class TraceCallbackHandler(BaseCallbackHandler):
    """每 run 一个实例，随 invoke/stream 的 config["callbacks"] 挂载。"""

    def __init__(
        self,
        store: MonitorStore,
        cfg: MonitoringConfig,
        thread_id: str,
        run_type: str = "unknown",
    ):
        super().__init__()
        self.store = store
        self.cfg = cfg
        self.thread_id = thread_id
        self.run_type = run_type
        self.run_id = uuid.uuid4().hex
        # 供 UI 实时读取的事件缓冲
        self.events: list[dict] = []
        self._node_seq = 0
        self._node_exec: dict[str, int] = {}  # 节点名 → 已执行次数（含已结束，供 retry_of 计数）
        self._had_node_error = False
        self._node_stack: list[dict] = []  # [{run_id, node, started_ns, started_at, retry_of}]
        self._llm_starts: dict[uuid.UUID, dict] = {}  # run_id -> {started_ns, model}
        self._token_totals = {"prompt": 0, "completion": 0}
        self._cost_total = 0.0
        self._root_open = False
        self._run_started_ns = time.perf_counter_ns()

    # ---- 通用工具 ----

    def _event(self, **payload: Any) -> None:
        self.events.append({"run_id": self.run_id, **payload})

    # ---- chain（图与节点）----

    def on_chain_start(
        self,
        serialized,
        inputs,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        name=None,
        **kwargs,
    ) -> None:
        try:
            name = name or (serialized or {}).get("name") or "?"
            if parent_run_id is None:
                # root run：整个图的执行
                self._root_open = True
                self._run_started_ns = time.perf_counter_ns()
                self.store.start_run(self.run_id, self.thread_id, self.run_type)
                self._event(type="run_start")
                return
            if name in KNOWN_NODES:
                # 同节点重复执行（重试）计数：含已结束的历史执行
                self._node_exec[name] = self._node_exec.get(name, 0)
                retry_of = self._node_exec[name]
                self._node_exec[name] += 1
                self._node_seq += 1
                self._node_stack.append(
                    {
                        "run_id": run_id,
                        "node": name,
                        "seq": self._node_seq,
                        "started_ns": time.perf_counter_ns(),
                        "started_at": _now(),
                        "retry_of": retry_of,
                    }
                )
                self._event(type="node_start", node=name, retry_of=retry_of)
        except Exception as exc:  # noqa: BLE001
            logger.debug("监控回调异常（on_chain_start）：%s", exc)

    def on_chain_end(self, outputs, *, run_id, parent_run_id=None, name=None, **kwargs) -> None:
        try:
            frame = self._pop_frame(run_id)
            if frame is None:
                if parent_run_id is None:
                    # 节点曾报错时，root 正常收尾也视为失败
                    self._finish_run(status="failure" if self._had_node_error else "success")
                return
            latency = (time.perf_counter_ns() - frame["started_ns"]) / 1e6
            self.store.write_span(
                self.run_id,
                frame["node"],
                frame["seq"],
                frame["started_at"],
                latency,
                "success",
                frame["retry_of"],
            )
            self._event(
                type="node_end",
                node=frame["node"],
                latency_ms=latency,
                status="success",
                retry_of=frame["retry_of"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("监控回调异常（on_chain_end）：%s", exc)

    def on_chain_error(self, error, *, run_id, parent_run_id=None, **kwargs) -> None:
        try:
            frame = self._pop_frame(run_id)
            msg = str(error)
            self._had_node_error = True
            if frame is not None:
                latency = (time.perf_counter_ns() - frame["started_ns"]) / 1e6
                self.store.write_span(
                    self.run_id,
                    frame["node"],
                    0,
                    frame["started_at"],
                    latency,
                    "failure",
                    frame["retry_of"],
                    msg,
                )
                self._event(
                    type="node_end",
                    node=frame["node"],
                    latency_ms=latency,
                    status="failure",
                    retry_of=frame["retry_of"],
                )
            else:
                self._finish_run(status="failure", error=msg)
        except Exception as exc:  # noqa: BLE001
            logger.debug("监控回调异常（on_chain_error）：%s", exc)

    def _pop_frame(self, run_id) -> dict | None:
        for i in range(len(self._node_stack) - 1, -1, -1):
            if self._node_stack[i]["run_id"] == run_id:
                frame = self._node_stack.pop(i)
                frame.setdefault("seq", 0)
                return frame
        return None

    def _finish_run(self, status: str, error: str | None = None) -> None:
        if not self._root_open:
            return
        self._root_open = False
        run_latency = (time.perf_counter_ns() - self._run_started_ns) / 1e6
        self.store.end_run(
            self.run_id,
            status,
            run_latency,
            self._token_totals["prompt"],
            self._token_totals["completion"],
            self._cost_total,
            error,
        )
        self._event(type="run_end", status=status, latency_ms=run_latency)
        try:
            self._evaluate_alerts(status, run_latency)
        except Exception as exc:  # noqa: BLE001
            logger.debug("告警评估异常：%s", exc)

    # ---- LLM 调用 ----

    def _on_model_start(self, run_id, serialized) -> None:
        model = "?"
        try:
            model = (serialized or {}).get("kwargs", {}).get("model") or "?"
        except Exception as exc:  # noqa: BLE001
            logger.debug("监控回调异常（解析模型名）：%s", exc)
        self._llm_starts[run_id] = {"started_ns": time.perf_counter_ns(), "model": model}

    def on_chat_model_start(
        self, serialized, messages, *, run_id, parent_run_id=None, **kwargs
    ) -> None:
        try:
            self._on_model_start(run_id, serialized)
        except Exception as exc:  # noqa: BLE001
            logger.debug("监控回调异常（on_chat_model_start）：%s", exc)

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs) -> None:
        try:
            self._on_model_start(run_id, serialized)
        except Exception as exc:  # noqa: BLE001
            logger.debug("监控回调异常（on_llm_start）：%s", exc)

    def on_chat_model_end(self, response, *, run_id, parent_run_id=None, **kwargs) -> None:
        self._on_model_end(run_id, response)

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs) -> None:
        self._on_model_end(run_id, response)

    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs) -> None:
        try:
            start = self._llm_starts.pop(run_id, None)
            latency = (time.perf_counter_ns() - start["started_ns"]) / 1e6 if start else 0.0
            node = self._current_node()
            self.store.write_llm_call(
                self.run_id,
                node,
                start["model"] if start else "?",
                None,
                None,
                None,
                latency,
                0.0,
                False,
                str(error),
            )
            self._event(type="llm_end", node=node, success=False)
        except Exception as exc:  # noqa: BLE001
            logger.debug("监控回调异常（on_llm_error）：%s", exc)

    def _on_model_end(self, run_id, response) -> None:
        try:
            start = self._llm_starts.pop(run_id, None)
            latency = (time.perf_counter_ns() - start["started_ns"]) / 1e6 if start else 0.0
            model = start["model"] if start else "?"
            # 模型名兜底：ChatResult.llm_output
            try:
                model = (response.llm_output or {}).get("model_name") or model
            except Exception as exc:  # noqa: BLE001
                logger.debug("监控回调异常（解析 llm_output）：%s", exc)

            usage = {}
            try:
                gen = response.generations[0]
                # 兼容历史嵌套结构：list[list[ChatGeneration]]
                if isinstance(gen, list):
                    gen = gen[0]
                msg = gen.message
                usage = msg.response_metadata.get("token_usage") or {}
                if not usage:
                    um = getattr(msg, "usage_metadata", None)
                    if um:
                        usage = {
                            "prompt_tokens": um.get("input_tokens"),
                            "completion_tokens": um.get("output_tokens"),
                            "total_tokens": um.get("total_tokens"),
                        }
            except Exception as exc:  # noqa: BLE001
                logger.debug("监控回调异常（解析 token 用量）：%s", exc)
            prompt = usage.get("prompt_tokens")
            completion = usage.get("completion_tokens")
            total = usage.get("total_tokens")

            cost = calc_cost(model, prompt, completion, self.cfg.pricing)
            if prompt is not None:
                self._token_totals["prompt"] += prompt
            if completion is not None:
                self._token_totals["completion"] += completion
            self._cost_total += cost

            node = self._current_node()
            self.store.write_llm_call(
                self.run_id, node, model, prompt, completion, total, latency, cost, True
            )
            self._event(
                type="llm_end",
                node=node,
                model=model,
                latency_ms=latency,
                prompt_tokens=prompt,
                completion_tokens=completion,
                cost_yuan=round(cost, 6),
                success=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("监控回调异常（on_llm_end）：%s", exc)

    def _current_node(self) -> str:
        return self._node_stack[-1]["node"] if self._node_stack else "unknown"

    # ---- 告警评估 ----

    def _evaluate_alerts(self, status: str, run_latency_ms: float) -> None:
        alert_cfg = self.cfg.alert
        spans = [e for e in self.events if e["type"] == "node_end"]

        # 1. 慢节点
        budgets = alert_cfg.node_latency_budget_ms
        for e in spans:
            budget = budgets.get(e["node"], budgets.get("default", 10000))
            if e["latency_ms"] > budget:
                detail = f"节点 {e['node']} 耗时 {e['latency_ms']:.0f}ms 超预算 {budget}ms"
                self.store.write_alert(self.run_id, "slow_node", "WARN", detail)
                logger.warning("[监控告警] %s", detail)

        # 2. 慢 run
        if run_latency_ms > alert_cfg.run_latency_budget_ms:
            detail = f"run 总耗时 {run_latency_ms:.0f}ms 超预算 {alert_cfg.run_latency_budget_ms}ms"
            self.store.write_alert(self.run_id, "slow_run", "WARN", detail)
            logger.warning("[监控告警] %s", detail)

        # 3. 失败
        if status == "failure":
            self.store.write_alert(self.run_id, "failure", "ERROR", "run 执行失败")
            logger.error("[监控告警] run %s 执行失败", self.run_id)

        # 4. 高重试
        retries = [e for e in spans if e.get("retry_of", 0) > alert_cfg.max_retries_per_run]
        for e in retries:
            detail = f"节点 {e['node']} 重复执行 {e['retry_of']} 次（阈值 {alert_cfg.max_retries_per_run}）"
            self.store.write_alert(self.run_id, "high_retry", "WARN", detail)
            logger.warning("[监控告警] %s", detail)

        # 5. 成功率过低
        stats = self.store.stats(alert_cfg.success_rate_window)
        if stats["success_rate"] is not None and stats["success_rate"] < alert_cfg.min_success_rate:
            detail = (
                f"近 {stats['run_count']} 次 run 成功率 {stats['success_rate']:.0%}"
                f"低于阈值 {alert_cfg.min_success_rate:.0%}"
            )
            self.store.write_alert(self.run_id, "low_success_rate", "WARN", detail)
            logger.warning("[监控告警] %s", detail)


def _now() -> str:
    from datetime import datetime

    return datetime.now().isoformat(timespec="milliseconds")
