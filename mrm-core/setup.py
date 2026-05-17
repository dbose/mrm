"""Setup script for MRM Core"""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="mrm-core",
    version="0.1.0",
    author="MRM Contributors",
    author_email="",
    description="Open source Model Risk Management CLI framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https/github.com/your-org/mrm-core",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        "typer[all]>=0.9.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
        "rich>=13.0.0",
        "jinja2>=3.1.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
    ],
    extras_require={
        "mlflow": ["mlflow>=2.8.0"],
        "ge": ["great-expectations>=0.18.0"],
        "wandb": ["wandb>=0.16.0"],
        "all": ["mlflow>=2.8.0", "great-expectations>=0.18.0", "wandb>=0.16.0"],
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mrm=mrm.cli.main:app",
        ],
    },
    include_package_data=True,
    package_data={
        "mrm": ["py.typed"],
    },
)
