#! /usr/bin/env python

from setuptools import setup
from setuptools import find_packages


try:
    with open('README.md', 'r') as f:
        readme = f.read()
except Exception:
    readme = str('')


install_requires = [
    'einops',
    'numpy',
    'matplotlib',
    'opencv-python',
    'pillow',
    'torch >= 2.0',
    'torchvision',
    'tqdm',
    'pytest',
    'igniter==0.0.4',
    'segment-anything',
]


setup(
    name='fsl',
    version='0.0.0',
    long_description=readme,
    packages=find_packages(),
    zip_safe=False,
    install_requires=install_requires,
    test_suite='tests',
)
