from __future__ import annotations
import asyncio
import time
from enum import Enum
from typing import Any, Awaitable, Callable, TypeVar
from loguru import logger
T = TypeVar("T")
class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"
class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max: int = 3,
        name: str = "default",
    ) -> None:
        self.failure_threshold = max(1, failure_threshold)
        self.recovery_timeout = max(0.1, recovery_timeout)
        self.half_open_max = max(1, half_open_max)
        self.name = name
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._half_open_attempts = 0
        self._lock = asyncio.Lock()
    @property
    def state(self) -> CircuitState:
        return self._state
    def _should_trip(self) -> bool:
        return self._failure_count >= self.failure_threshold
    def _should_attempt_reset(self) -> bool:
        if self._last_failure_time is None:
            return True
        return (time.monotonic() - self._last_failure_time) >= self.recovery_timeout
    def _record_success(self) -> None:
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._half_open_attempts = 0
        self._last_failure_time = None
    def _record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._half_open_attempts = 0
            logger.warning(
                "熔断器 [{}] Half-Open 试探失败，重新打开 circuit",
                self.name,
            )
        elif self._should_trip():
            self._state = CircuitState.OPEN
            logger.error(
                "熔断器 [{}] 已达失败阈值 {}，状态切换为 OPEN",
                self.name,
                self.failure_threshold,
            )
    async def call(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        async with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_attempts = 0
                    logger.info("熔断器 [{}] 进入 Half-Open 试探", self.name)
                else:
                    raise RuntimeError(
                        f"熔断器 [{self.name}] 处于 OPEN 状态，拒绝调用"
                    )
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_attempts >= self.half_open_max:
                    raise RuntimeError(
                        f"熔断器 [{self.name}] Half-Open 试探次数已达上限"
                    )
                self._half_open_attempts += 1
        try:
            result = await func(*args, **kwargs)
        except Exception as exc:
            async with self._lock:
                self._record_failure()
            logger.exception("熔断器 [{}] 包裹调用失败: {}", self.name, exc)
            raise
        async with self._lock:
            self._record_success()
        return result
