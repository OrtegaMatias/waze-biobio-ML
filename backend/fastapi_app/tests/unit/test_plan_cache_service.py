from backend.fastapi_app.app.services.plan_cache_service import PlanResultCache


def test_cache_returns_an_independent_copy():
    cache = PlanResultCache[dict](ttl_seconds=60)
    cache.set("route", {"routes": ["original"]})

    first = cache.get("route")
    first["routes"].append("changed")

    assert cache.get("route") == {"routes": ["original"]}


def test_cache_expires_entries():
    now = [100.0]
    cache = PlanResultCache[str](ttl_seconds=10, clock=lambda: now[0])
    cache.set("route", "result")

    now[0] = 109.9
    assert cache.get("route") == "result"
    now[0] = 110.0
    assert cache.get("route") is None


def test_cache_evicts_the_least_recently_used_entry():
    cache = PlanResultCache[str](max_entries=2, ttl_seconds=60)
    cache.set("first", "one")
    cache.set("second", "two")
    assert cache.get("first") == "one"

    cache.set("third", "three")

    assert cache.get("second") is None
    assert cache.get("first") == "one"
    assert cache.get("third") == "three"

