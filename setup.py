# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path
from setuptools import setup, find_packages


def parse_requirements_file(path: str = "requirements.txt"):
    return [line.rstrip() for line in open(path, "r")]


with open("README.md", "r") as f:
    long_description = f.read()


init_str = Path("mbrl/__init__.py").read_text()
version = init_str.split("__version__ = ")[1].rstrip().strip('"')

setup(
    name="mbrl",
    version=version,
    author="Facebook AI Research",
    description="A PyTorch library for model-based reinforcement learning research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/facebookresearch/mbrl-lib",
    packages=find_packages(),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=parse_requirements_file(),
    include_package_data=True,
    python_requires=">=3.8",
    zip_safe=False,
)
