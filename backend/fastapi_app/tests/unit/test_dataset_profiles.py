from backend.fastapi_app.app.core import dataset
from algorithms.recommenders import data_loader


def test_available_profiles_include_concepcion():
    profile_keys = {key for key, _ in dataset.available_profiles()}
    assert "concepcion" in profile_keys


def test_set_profile_switches_ratings_path():
    original = dataset.get_profile()
    try:
        dataset.set_profile("concepcion")
        assert data_loader.get_user_ratings_path().name == "user_ratings_concepcion.csv"
    finally:
        dataset.set_profile(original)
