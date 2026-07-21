from __future__ import annotations

import time
from collections import OrderedDict
from copy import deepcopy
from threading import Lock
from typing import Callable, Generic, Hashable, TypeVar

T = TypeVar("T")


class PlanResultCache(Generic[T]):
    """Small in-memory LRU cache with per-entry expiration."""

    def __init__(
        self,
        max_entries: int = 32,
        ttl_seconds: float = 900.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_entries = max(1, int(max_entries))
        self._ttl_seconds = max(0.0, float(ttl_seconds))
        self._clock = clock
        self._entries: OrderedDict[Hashable, tuple[float, T]] = OrderedDict()
        self._lock = Lock()

    def get(self, key: Hashable) -> T | None:
        now = self._clock()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at <= now:
                self._entries.pop(key, None)
                return None
            self._entries.move_to_end(key)
            return deepcopy(value)

    def set(self, key: Hashable, value: T) -> None:
        expires_at = self._clock() + self._ttl_seconds
        with self._lock:
            self._entries[key] = (expires_at, deepcopy(value))
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

