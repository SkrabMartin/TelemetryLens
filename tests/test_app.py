from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.mark.slow
def test_app_renders_demo_without_errors():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    assert not at.error
    labels = {m.label: m.value for m in at.metric}
    assert labels["Machines"] == "5"
    assert labels["Duplicates"] == "20"
    assert len(at.tabs) == 6
