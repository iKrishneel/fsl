#! /usr/bin/env python

from setuptools import find_packages, setup

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
    'igniter==1.0.0',
]

setup(
    name='fsl',
    version='0.0.1',
    long_description=readme,
    packages=find_packages(),
    zip_safe=False,
    install_requires=install_requires,
    test_suite='tests',
    extras_require={
        'full': [
            'clip @ git+https://github.com/openai/CLIP.git',
            'timm',
            'segment-anything',
            'ultralytics==8.1',
            'fast_pytorch_kmeans',
        ],
        'dev': [
            'jupyterlab',
        ],
    },
)
