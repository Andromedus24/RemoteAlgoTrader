#!/usr/bin/env python3
"""
RemoteAlgoTrader v1.0 Setup Script
Installation and setup utilities for the AI-powered trading bot
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="remote-algo-trader",
    version="1.0.0",
    author="RemoteAlgoTrader Team",
    author_email="your.email@example.com",
    description="AI-Powered Algorithmic Trading Bot with Sentiment Analysis & Technical Indicators",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/RemoteAlgoTrader",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "optional": [
            "websocket-client>=1.6.0",
            "plotly>=5.15.0",
            "streamlit>=1.25.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "remote-algo-trader=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="trading, algorithmic-trading, ai, sentiment-analysis, technical-analysis, finance, investment",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/RemoteAlgoTrader/issues",
        "Source": "https://github.com/yourusername/RemoteAlgoTrader",
        "Documentation": "https://github.com/yourusername/RemoteAlgoTrader#readme",
    },
)
