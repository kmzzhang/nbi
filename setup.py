from setuptools import setup, find_packages
import os

PACKAGES = find_packages(where="src")

INSTALL_REQUIRES = [
    "torch>=1.4",
    "numpy",
    "scipy",
    "matplotlib",
    "corner",
    "joblib",
    "pytest",
    "tqdm",
    'wandb',
    'multiprocess',
    'ipywidgets'
]

setup(
    name='nbi',
    author='Keming Zhang',
    author_email='kemingz@berkeley.edu',
    packages=PACKAGES,
    package_dir={"": "src"},
    version='0.0.1',
    install_requires=INSTALL_REQUIRES
)
