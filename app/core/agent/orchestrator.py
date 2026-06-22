from __future__ import annotations
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Protocol, Sequence
from app.core.agent.planner import PlannerAgent
from app.core.agent.react_agent import AgentResult, ReActAgent
from app.core.agent.reflection import ReflectionAgent, ReflectionReport
logger = logging.getLogger(__name__)
OrchestrationMode = Literal["react", "plan_execute"]
class OrchestratorConfig(Protocol):
    def get(self, key: str, default: Any = None) -> Any:
        ...
class ModelRouter(Protocol):
    def get_llm(self, purpose: str) -> Any:
        ...
class MemoryManager(Protocol):
    async def get_relevant(self, session_id: str, query: str, limit: int = 8) -> List[str]:
        ...
    async def append_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        ...
class ToolRegistry(Protocol):
    def list_tool_names(self) -> List[str]:
        ...
    async def invoke(self, name: str, arguments: Dict[str, Any]) -> str:
        ...
class Tracer(Protocol):
    def new_trace_id(self) -> str:
        ...
    def start_span(self, name: str, trace_id: str, attributes: Optional[Dict[str, Any]] = None) -> Any:
        ...
    def end_span(self, span: Any, error: Optional[BaseException] = None) -> None:
        ...
    def log_event(self, trace_id: str, name: str, payload: Dict[str, Any]) -> None:
        ...
@dataclass
class IntentContext:
    intent: str = "general"
    confidence: float = 1.0
    slots: Dict[str, Any] = field(default_factory=dict)
    preferred_mode: Optional[OrchestrationMode] = None
    allowed_tools: Optional[List[str]] = None
@dataclass
class AgentResponse:
    answer: str
    mode_used: OrchestrationMode
    success: bool
    trace_id: str
    intent: IntentContext
    steps: List[Dict[str, Any]] = field(default_factory=list)
    reflection: Optional[ReflectionReport] = None
    error: Optional[str] = None
    degraded: bool = False
class _LLMAdapter:
    def __init__(self, llm: Any) -> None:
        self._llm = llm
    async def acomplete(self, messages: Sequence[Dict[str, str]], **kwargs: Any) -> str:
        fn = getattr(self._llm, "acomplete", None)
        if callable(fn):
            return await fn(messages, **kwargs)
        fn2 = getattr(self._llm, "complete", None)
        if callable(fn2):
            import asyncio
            return await asyncio.to_thread(fn2, messages, **kwargs)
        raise TypeError("LLM 需实现 acomplete 或 complete")
class AgentOrchestrator:
    def __init__(
        self,
        config: OrchestratorConfig,
        model_router: ModelRouter,
        memory_manager: MemoryManager,
        tool_registry: ToolRegistry,
        tracer: Tracer,
    ) -> None:
        self._config = config
        self._model_router = model_router
        self._memory = memory_manager
        self._tools = tool_registry
        self._tracer = tracer
        self._enable_reflection = bool(config.get("enable_reflection", True))
        self._fallback_on_plan_failure = bool(config.get("fallback_react_on_plan_failure", True))
    def _tool_names(self, intent: IntentContext) -> List[str]:
        all_names = self._tools.list_tool_names()
        if intent.allowed_tools:
            return [n for n in intent.allowed_tools if n in all_names]
        return all_names
    async def run(
        self,
        user_input: str,
        session_id: str,
        mode: str = "react",
        intent: Optional[IntentContext] = None,
    ) -> AgentResponse:
        intent_ctx = intent or IntentContext()
        if intent_ctx.preferred_mode:
            mode = intent_ctx.preferred_mode        
        trace_id = self._tracer.new_trace_id()
        span = self._tracer.start_span(
            "orchestrator.run",
            trace_id,
            attributes={"session_id": session_id, "mode": mode, "intent": intent_ctx.intent},
        )
        self._tracer.log_event(
            trace_id,
            "orchestrator.start",
            {"user_input_len": len(user_input), "mode": mode},
        )
        steps: List[Dict[str, Any]] = []
        degraded = False
        error_msg: Optional[str] = None
        result: Optional[AgentResult] = None
        mode_used: OrchestrationMode = "react"
        try:
            if mode == "plan_execute":
                mode_used = "plan_execute"
                result = await self._run_plan_execute(
                    user_input,
                    session_id,
                    intent_ctx,
                    trace_id,
                    steps,
                )
                if not result.success and self._fallback_on_plan_failure:
                    self._tracer.log_event(trace_id, "orchestrator.fallback", {"to": "react"})
                    degraded = True
                    result = await self._run_react(
                        user_input,
                        session_id,
                        intent_ctx,
                        trace_id,
                        steps,
                        suffix="[降级] ",
                    )
                    mode_used = "react"
            else:
                result = await self._run_react(
                    user_input,
                    session_id,
                    intent_ctx,
                    trace_id,
                    steps,
                )
            answer = (result.final_answer if result else "") or ""
            success = bool(result and result.success)
            reflection_report: Optional[ReflectionReport] = None
            if self._enable_reflection and answer:
                try:
                    reflection_report = await self._run_reflection(
                        user_input,
                        answer,
                        trace_id,
                        steps,
                    )
                except Exception as e:                
                    logger.warning("反思阶段失败，已跳过: %s", e)
                    self._tracer.log_event(trace_id, "reflection.skipped", {"error": str(e)})
            resp = AgentResponse(
                answer=answer,
                mode_used=mode_used,
                success=success,
                trace_id=trace_id,
                intent=intent_ctx,
                steps=steps,
                reflection=reflection_report,
                error=result.error if result and not success else None,
                degraded=degraded,
            )
            self._tracer.end_span(span, error=None)
            return resp
        except Exception as e:                
            logger.exception("编排器未捕获异常")
            error_msg = str(e)
            self._tracer.log_event(trace_id, "orchestrator.error", {"error": error_msg})
            self._tracer.end_span(span, error=e)
            return AgentResponse(
                answer="",
                mode_used="react",
                success=False,
                trace_id=trace_id,
                intent=intent_ctx,
                steps=steps,
                error=error_msg,
                degraded=degraded,
            )
    async def _run_react(
        self,
        user_input: str,
        session_id: str,
        intent: IntentContext,
        trace_id: str,
        steps: List[Dict[str, Any]],
        suffix: str = "",
    ) -> AgentResult:
        llm = _LLMAdapter(self._model_router.get_llm("react"))
        max_steps = int(self._config.get("react_max_steps", 10))
        async def trace_cb(rec: Dict[str, Any]) -> None:
            payload = {"ts": time.time(), **rec}
            steps.append(payload)
            self._tracer.log_event(trace_id, "react.step", payload)
        agent = ReActAgent(
            llm=llm,
            tools=self._tools,
            memory=self._memory,
            max_steps=max_steps,
            session_id=session_id,
        )
        ctx: Dict[str, Any] = {
            "session_id": session_id,
            "tool_names": self._tool_names(intent),
            "trace_callback": trace_cb,
            "extra_system": suffix + f"当前识别意图：{intent.intent}，槽位：{intent.slots}",
        }
        self._tracer.log_event(trace_id, "react.begin", {})
        result = await agent.run(user_input, ctx)
        self._tracer.log_event(
            trace_id,
            "react.end",
            {"success": result.success, "steps": len(result.steps)},
        )
        return result
    async def _run_plan_execute(
        self,
        user_input: str,
        session_id: str,
        intent: IntentContext,
        trace_id: str,
        steps: List[Dict[str, Any]],
    ) -> AgentResult:
        llm = _LLMAdapter(self._model_router.get_llm("planner"))
        async def trace_cb(payload: Dict[str, Any]) -> None:
            entry = {"ts": time.time(), **payload}
            steps.append(entry)
            self._tracer.log_event(trace_id, "plan_execute", entry)
        planner = PlannerAgent(
            llm=llm,
            tools=self._tools,
            memory=self._memory,
            max_replan_attempts=int(self._config.get("max_replan_attempts", 2)),
        )
        self._tracer.log_event(trace_id, "planner.begin", {})
        result = await planner.run_with_replan(
            user_input,
            session_id,
            tool_names=self._tool_names(intent),
            trace_callback=trace_cb,
        )
        self._tracer.log_event(
            trace_id,
            "planner.end",
            {"success": result.success, "steps": len(result.steps)},
        )
        return result
    async def _run_reflection(
        self,
        user_query: str,
        answer: str,
        trace_id: str,
        steps: List[Dict[str, Any]],
    ) -> ReflectionReport:
        llm = _LLMAdapter(self._model_router.get_llm("reflection"))
        agent = ReflectionAgent(llm=llm, min_quality_to_pass=int(self._config.get("reflection_min_quality", 60)))
        trace_summary = str([s.get("phase") or s.get("subtask_id") for s in steps[-20:]])
        self._tracer.log_event(trace_id, "reflection.begin", {})
        report = await agent.reflect(
            user_query,
            answer,
            evidence_snippets=None,
            trace_summary=trace_summary,
        )
        self._tracer.log_event(
            trace_id,
            "reflection.end",
            {"quality": report.quality_score, "hallucination": report.likely_hallucination},
        )
        return report
class InMemoryTracer:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
    def new_trace_id(self) -> str:
        return str(uuid.uuid4())
    def start_span(self, name: str, trace_id: str, attributes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {"name": name, "trace_id": trace_id, "attributes": attributes or {}}
    def end_span(self, span: Any, error: Optional[BaseException] = None) -> None:
        self.events.append({"type": "end_span", "span": span, "error": str(error) if error else None})
    def log_event(self, trace_id: str, name: str, payload: Dict[str, Any]) -> None:
        self.events.append({"trace_id": trace_id, "name": name, "payload": payload})
