from distutils.core import setup
from pip.req import parse_requirements
from pathlib import Path

from setuptools import find_packages

setup(
    name="deepul",
    version="0.1.0",
    packages=find_packages(),
    license="MIT License",
    install_requires=[
        str(req.req)
        for req in parse_requirements(Path("requirements.txt"), session=False)
        if req.req
        and not req.req.startswith("-e")  # Exclude editable installs
        and not req.req.startswith("git+")  # Exclude git installs
    ],
)
