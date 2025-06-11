from setuptools import setup, find_packages

setup(
    name="mose",
    version="0.1",
    packages=find_packages(include=["config", "data", "strategy", "utils", "monitor", "trading_system", "strategy_optimizer", "advisor", "analysis", "backtest", "examples", "scripts", "results", "tests", "updateNSSInfos", "bak"]),
    install_requires=[
        # 依赖已经在 requirements.txt 中定义
    ],
)