from setuptools import setup, find_packages

setup(
    name="mose",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "xgboost>=1.5.0",
        "stable-baselines3>=1.5.0",
        "gym>=0.21.0",
        "ta-lib>=0.4.24"
    ],
)