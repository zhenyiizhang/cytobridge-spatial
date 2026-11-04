"""Compatibility shim for tooling that still invokes ``setup.py`` directly.

All project metadata, including the dynamically resolved version, is declared in
``pyproject.toml``.  Keeping this file argument-free prevents a second metadata
source from drifting out of sync.
"""

from setuptools import setup


setup()
