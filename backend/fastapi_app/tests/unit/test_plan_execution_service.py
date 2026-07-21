import asyncio
import threading
import time

import pytest

from algorithms.recommenders.routing import RouteSearchCancelled
from backend.fastapi_app.app.services.plan_execution_service import PlanExecutionCoordinator


def test_duplicate_requests_share_one_computation():
    async def scenario():
        coordinator = PlanExecutionCoordinator[str]()
        calls = 0

        def compute(_should_cancel):
            nonlocal calls
            calls += 1
            return "route"

        first = await coordinator.acquire("same", compute)
        second = await coordinator.acquire("same", compute)

        assert await first.task == "route"
        assert await second.task == "route"
        assert calls == 1
        await first.release()
        await second.release()

    asyncio.run(scenario())


def test_different_requests_never_compute_simultaneously():
    async def scenario():
        coordinator = PlanExecutionCoordinator[str]()
        active = 0
        maximum_active = 0
        lock = threading.Lock()

        def compute(_should_cancel):
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.03)
            with lock:
                active -= 1
            return "route"

        first = await coordinator.acquire("first", compute)
        second = await coordinator.acquire("second", compute)
        assert await asyncio.gather(first.task, second.task) == ["route", "route"]
        assert maximum_active == 1
        await first.release()
        await second.release()

    asyncio.run(scenario())


def test_last_cancelled_waiter_stops_the_shared_computation():
    async def scenario():
        coordinator = PlanExecutionCoordinator[str]()
        started = threading.Event()

        def compute(should_cancel):
            started.set()
            while not should_cancel():
                time.sleep(0.005)
            raise RouteSearchCancelled()

        lease = await coordinator.acquire("cancel", compute)
        await asyncio.to_thread(started.wait, 1)
        await lease.release(cancelled=True)

        with pytest.raises(RouteSearchCancelled):
            await lease.task

    asyncio.run(scenario())
