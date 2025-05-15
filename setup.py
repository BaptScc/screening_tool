"""
from setuptools import setup, find_packages

setup(
    name="papermill_screener",
    version="0.1.0",
    author="Baptiste Scancar",
    author_email="baptiste.scancar@agrocampus-ouest.fr",
    description="Package dédié au screening de texte frauduleux",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "pandas", "torch", "transformers", "nltk", "tqdm"
    ],
    python_requires='>=3.7',
    entry_points={
        "console_scripts": [
            "mon-analyse=mon_package.processing:main"
        ],
    },
)
"""
