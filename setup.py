"""Installation script for the 'leapsim' python package."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from setuptools import setup, find_packages

import os

root_dir = os.path.dirname(os.path.realpath(__file__))


# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # RL
    "gym",
    "torch",
    "omegaconf",
    "termcolor",
    "hydra-core>=1.1",
    "rl-games==1.5.2",
    "pyvirtualdisplay",
    ]



# Installation operation
setup(
    name="leapsim",
    author="Ananye Agarwal",
    version="1.0.0",
    description="LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand simulation environment",
    keywords=["robotics", "rl"],
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    packages=find_packages("."),
    classifiers=["Natural Language :: English", "Programming Language :: Python :: 3.7, 3.8"],
    zip_safe=False,
)

# EOF
