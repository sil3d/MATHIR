"""
Setup script pour installer mathir_lib comme package Python
"""

from setuptools import setup, find_packages

with open("README_LIB.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mathir-lib",
    version="2.0.0",
    author="MATHIR Team",
    author_email="contact@mathir.ai",
    description="Memory-Augmented Transformer with Hierarchical Retention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mathir-project/mathir",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "benchmark": [
            "streamlit>=1.28.0",
            "plotly>=5.17.0",
            "pandas>=2.0.0",
        ],
    },
    keywords="deep-learning reinforcement-learning memory transformer autonomous-driving",
    project_urls={
        "Bug Reports": "https://github.com/mathir-project/mathir/issues",
        "Source": "https://github.com/mathir-project/mathir",
        "Documentation": "https://mathir.readthedocs.io",
    },
)
