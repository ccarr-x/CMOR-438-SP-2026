"""Setuptools shim so older pip can run editable installs from pyproject.toml."""

from setuptools import setup

if __name__ == "__main__":
    setup()
