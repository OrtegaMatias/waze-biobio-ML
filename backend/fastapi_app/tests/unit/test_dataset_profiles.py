from algorithms.recommenders import data_loader
import pandas as pd

from algorithms.recommenders.geo_profiles import filter_dataframe_for_profile
from backend.fastapi_app.app.core import dataset


def test_available_profiles_only_includes_gran_concepcion():
    assert dataset.available_profiles() == [("gran_concepcion", "Gran Concepcion")]


def test_default_profile_is_gran_concepcion():
    assert dataset.get_profile() == "gran_concepcion"
    assert data_loader.get_data_profile() == "gran_concepcion"


def test_set_profile_rejects_other_profiles():
    try:
        dataset.set_profile("otro_perfil")
    except ValueError as exc:
        assert "gran_concepcion" in str(exc)
    else:
        raise AssertionError("Expected unsupported profile to fail")


def test_profile_filter_accepts_osm_place_labels():
    df = pd.DataFrame(
        {
            "comuna": [
                "Concepcion, Region del Biobio, Chile",
                "San Pedro de la Paz, Region del Biobio, Chile",
                "Los Angeles, Region del Biobio, Chile",
            ],
            "value": [1, 2, 3],
        }
    )

    filtered = filter_dataframe_for_profile(df, "gran_concepcion")

    assert filtered["value"].tolist() == [1, 2]
