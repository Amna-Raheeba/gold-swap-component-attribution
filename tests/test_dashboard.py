"""Tests for the Streamlit dashboard (AppTest)."""

import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def test_dashboard_renders():
    app = AppTest.from_file(APP)
    app.run(timeout=120)
    assert not app.exception
    assert "Gold-Swap Component Attribution" in app.title[0].value


def test_dashboard_tabs_present():
    app = AppTest.from_file(APP)
    app.run(timeout=120)
    assert not app.exception
    tab_labels = [t.label for t in app.tabs]
    assert tab_labels == ["Attribution", "Fix Order", "Evaluation"]


def test_dashboard_switching_system():
    app = AppTest.from_file(APP)
    app.run(timeout=120)
    assert not app.exception
    app.sidebar.selectbox[0].set_value("phm").run()
    assert not app.exception
