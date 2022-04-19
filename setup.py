from setuptools import setup, find_packages
import os

PACKAGES = find_packages(where="src")

INSTALL_REQUIRES = [
    "torch>=1.4",
    "numpy",
    "matplotlib",
    "corner",
    "joblib",
    "pytest",
    "tqdm"
]

setup(
    name='nbi',
    author='Keming Zhang',
    author_email='kemingz@berkeley.edu',
    packages=PACKAGES,
    package_dir={"": "src"},
    install_requires=INSTALL_REQUIRES
)
