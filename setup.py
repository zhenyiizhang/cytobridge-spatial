"""Compatibility installer for environments with pre-PEP-621 setuptools."""

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


def _requirements() -> list[str]:
    return [
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


setup(
    name="CytoBridge",
    version="0.1.0",
    description=(
        "CytoBridge package for spatiotemporal single-cell and spatial "
        "transcriptomics analysis."
    ),
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["CytoBridge", "CytoBridge.*"]),
    include_package_data=True,
    package_data={"CytoBridge": ["configs/*.yaml"]},
    install_requires=_requirements(),
    python_requires=">=3.10",
)
