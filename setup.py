from setuptools import setup, find_packages

setup(
    name="DGSA_Light",                     # your package name
    version="0.1.0",                       # version number
    author="David Zhen",         # optional
    description="Lightweight DGSA implementation for GCS importance sampling workflow",
    packages=find_packages(
        include=["DGSA_Light", "DGSA_Light.*"]
    ),                                     # automatically includes submodules
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "scipy"
    ],                                     # dependencies used by your code
    python_requires=">=3.8",
)
