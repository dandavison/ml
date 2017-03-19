import os

from setuptools import find_packages
from setuptools import setup


setup(
    name='ml',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
)
