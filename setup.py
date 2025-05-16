from setuptools import setup, find_packages

setup(
    name="screening_tool",  # Doit correspondre à ton dossier de package
    version="0.1.0",
    author="Baptiste Scancar",
    author_email="baptiste.scancar@agrocampus-ouest.fr",
    description="Package dédié au screening de texte frauduleux",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "pandas == 2.2.3",
        "torch == 2.5.1",
        "transformers == 4.49.0",
        "nltk == 3.9.1",
        "tqdm == 4.67.1"
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "mon-analyse=screening_tool.processing:main"
        ],
    },
)
