from setuptools import setup, find_packages

setup(
    name="Spider",
    version="0.1",
    packages=find_packages(),
    install_requires=['scikit-learn==1.1.3', 'numpy==1.23.4'],
)