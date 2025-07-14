from setuptools import setup, find_packages

setup(
    name="intelligent-support-agent",
    version="1.0.0",
    description="Intelligent Customer Support Agent - Phase 1",
    author="Muhammad Huzaifa Khan",
    author_email="themuhammadhuzaifakhan@gmail.com",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
