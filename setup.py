"""LoRAven 安装脚本
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# 获取项目根目录
HERE = Path(__file__).parent

# 读取 README 文件
def read_readme():
    readme_path = HERE / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as fh:
            return fh.read()
    return "LoRAven - Adaptive Dynamic Low-Rank Neural Systems"

# 读取 requirements.txt
def read_requirements():
    req_path = HERE / "requirements.txt"
    if req_path.exists():
        with open(req_path, "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    return ["torch>=1.9.0", "numpy>=1.20.0", "matplotlib>=3.3.0", "pyyaml>=5.4.0"]

# 读取版本信息
def get_version():
    version_file = HERE / "__init__.py"
    if version_file.exists():
        with open(version_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"

setup(
    name="loraven",
    version=get_version(),
    author="LoRAven Team",
    author_email="loraven@example.com",
    description="Adaptive Dynamic Low-Rank Neural Systems - 自适应动态低秩神经网络系统",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/loraven/loraven",
    packages=["loraven", "loraven.models", "loraven.schedulers", "loraven.trainers", "loraven.utils"],
    package_dir={"loraven": "."},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "loraven-train=loraven.trainers.train_loraven:main",
            "loraven-test=loraven.tests.unit_tests:main",
        ],
    },
    include_package_data=True,
    package_data={
        "loraven": [
            "*.yaml",
            "*.yml",
            "*.json",
            "*.md",
            "experiments/*.yaml",
            "experiments/*.yml",
            "experiments/*.sh",
            "configs/*.yaml",
            "configs/*.yml",
            "data/*.json",
            "models/*.pth",
            "checkpoints/*.pth",
        ],
    },
    zip_safe=False,
)
