# Copyright (c) 2023, 2026 Francesco Orabona
# Licensed under the MIT License. See LICENSE in the project root.

from setuptools import setup

# To use a consistent encoding
from codecs import open
from os import path
import re
from urllib.parse import urljoin, urlsplit

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))
PROJECT_URL = 'https://github.com/bremen79/parameterfree'

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


def _absolute_readme_link(match):
    prefix, target, suffix = match.groups()
    if urlsplit(target).scheme or target.startswith(('#', '//')):
        return match.group(0)
    base = ('https://raw.githubusercontent.com/bremen79/parameterfree/main/'
            if prefix.startswith('!') else PROJECT_URL + '/blob/main/')
    return prefix + urljoin(base, target.lstrip('/')) + suffix


# GitHub resolves relative README links; PyPI needs absolute file/image URLs.
long_description = re.sub(r'(!?\[[^\]]*\]\()([^\s)]+)(\))',
                          _absolute_readme_link, long_description)

# This call to setup() does all the work
setup(
    name="parameterfree",
    version="0.0.2",
    description="Parameter-Free Optimizers for PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=PROJECT_URL,
    author="Francesco Orabona",
    author_email="francesco@orabona.com",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.6',
    packages=["parameterfree"],
    install_requires=["torch>=1.9"],
    include_package_data=True,
)
