# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import time
from typing import Any, Dict

import requests


DEFAULT_SCENARIO = {
    "origin": {"lat": -36.8267, "lon": -73.0498},
    "destination": {"lat": -36.8114, "lon": -73.0490},
    "preferences": [],
    "ubcf_preferences": [],
    "ibcf_preferences": [],
    "day_of_week": "Wednesday",
    "departure_hour": 8.0,
    "avoid_congestion": True,
    "avoid_accidents": False,
}


def wait_until_ready(base_url: str, timeout_seconds: int) -> Dict[str, Any]:
    start = time.perf_counter()
    while time.perf_counter() - start < timeout_seconds:
        response = requests.get(f"{base_url}/readyz", timeout=10)
        response.raise_for_status()
        body = response.json()
        if body.get("ready"):
            return body
        time.sleep(2)
    raise TimeoutError("El backend no quedó listo dentro del tiempo esperado.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark simple del flujo principal de la demo.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--wait-ready", action="store_true", help="Espera a que /readyz esté listo antes de medir.")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    session = requests.Session()
    session.post(f"{args.base_url}/system/bootstrap", json={}, timeout=15).raise_for_status()

    if args.wait_ready:
        ready = wait_until_ready(args.base_url, args.timeout)
        bootstrap = ready.get("bootstrap") or {}
        print(
            "readyz",
            f"status={ready.get('status')}",
            f"duration_ms={bootstrap.get('duration_ms')}",
            f"nodes={bootstrap.get('routing_nodes')}",
            f"segments={bootstrap.get('routing_segments')}",
        )

    started = time.perf_counter()
    response = session.post(
        f"{args.base_url}/routes/optimal",
        json=DEFAULT_SCENARIO,
        timeout=args.timeout,
    )
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    body = response.json()
    print(
        "routes/optimal",
        f"status={response.status_code}",
        f"elapsed_s={elapsed:.3f}",
        f"best_balance={body['comparison']['best_balance_variant']}",
        f"fastest={body['comparison']['fastest_variant']}",
    )


if __name__ == "__main__":
    main()
