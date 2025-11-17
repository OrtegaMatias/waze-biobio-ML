from types import SimpleNamespace

from backend.fastapi_app.app.schemas.recommendations import PlaygroundRequest
from backend.fastapi_app.app.services.recommendation_service import RecommendationService


class DummyModel:
    def __init__(self, tag: str) -> None:
        self.tag = tag

    def recommend(self, user_id: str, known_vias, top_n: int):
        return [
            SimpleNamespace(via=f"{self.tag}-via", estimated_rating=4.5, strategy=self.tag)
        ]


def _service_with_dummies() -> RecommendationService:
    service = RecommendationService()
    service._cf_models = {
        "ubcf": DummyModel("ubcf"),
        "ibcf": DummyModel("ibcf"),
    }
    return service


def test_playground_returns_only_requested_strategies():
    service = _service_with_dummies()
    payload = PlaygroundRequest(user_id="user-1", known_vias=["Ruta Azul"], limit=2, strategies=["ibcf"])

    results = service.playground_recommendations(payload)

    assert "ibcf" in results and "ubcf" not in results
    assert results["ibcf"][0].via == "ibcf-via"


def test_playground_defaults_to_both_strategies():
    service = _service_with_dummies()
    payload = PlaygroundRequest(user_id="user-2", known_vias=[], limit=1, strategies=[])

    results = service.playground_recommendations(payload)

    assert set(results.keys()) == {"ubcf", "ibcf"}
    assert results["ubcf"][0].strategy == "ubcf"
