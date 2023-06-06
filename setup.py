from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# This call to setup() does all the work
setup(
    name="parameterfree",
    version="0.0.1",
    description="Parameter-Free Optimizers for Pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bremen79/parameterfree",
    author="Francesco Orabona",
    author_email="francesco@orabona.com",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.6',
    packages=["parameterfree"],
    include_package_data=True,
)
