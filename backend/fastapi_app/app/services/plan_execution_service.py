from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Awaitable, Callable, Generic, Hashable, TypeVar

T = TypeVar("T")
Compute = Callable[[Callable[[], bool]], T]


@dataclass
class _Execution(Generic[T]):
    task: asyncio.Task[T]
    cancellation: threading.Event
    waiters: int = 0


class PlanExecutionLease(Generic[T]):
    def __init__(
        self,
        coordinator: "PlanExecutionCoordinator[T]",
        key: Hashable,
        execution: _Execution[T],
    ) -> None:
        self._coordinator = coordinator
        self._key = key
        self._execution = execution
        self._released = False

    @property
    def task(self) -> asyncio.Task[T]:
        return self._execution.task

    async def release(self, cancelled: bool = False) -> None:
        if self._released:
            return
        self._released = True
        await self._coordinator._release(self._key, self._execution, cancelled)


class PlanExecutionCoordinator(Generic[T]):
    """Serializes expensive plans and shares work for identical requests."""

    def __init__(self) -> None:
        self._execution_lock = asyncio.Lock()
        self._capacity = asyncio.Semaphore(1)
        self._executions: dict[Hashable, _Execution[T]] = {}

    async def acquire(self, key: Hashable, compute: Compute[T]) -> PlanExecutionLease[T]:
        async with self._execution_lock:
            execution = self._executions.get(key)
            if execution is None:
                cancellation = threading.Event()
                task = asyncio.create_task(self._run(compute, cancellation))
                execution = _Execution(task=task, cancellation=cancellation)
                self._executions[key] = execution
            execution.waiters += 1
            return PlanExecutionLease(self, key, execution)

    async def _run(self, compute: Compute[T], cancellation: threading.Event) -> T:
        async with self._capacity:
            return await asyncio.to_thread(compute, cancellation.is_set)

    async def _release(
        self,
        key: Hashable,
        execution: _Execution[T],
        cancelled: bool,
    ) -> None:
        async with self._execution_lock:
            execution.waiters = max(0, execution.waiters - 1)
            if execution.waiters > 0:
                return
            if cancelled and not execution.task.done():
                execution.cancellation.set()
            if self._executions.get(key) is execution:
                self._executions.pop(key, None)
            if not execution.task.done():
                execution.task.add_done_callback(self._consume_unobserved_result)

    @staticmethod
    def _consume_unobserved_result(task: asyncio.Task[T]) -> None:
        if not task.cancelled():
            task.exception()

