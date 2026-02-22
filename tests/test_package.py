from __future__ import annotations

import importlib.metadata

import gammaspy as m


def test_version():
    assert importlib.metadata.version("gammaspy") == m.__version__
