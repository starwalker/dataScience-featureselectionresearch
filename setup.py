from distutils.core import setup
from setuptools import find_packages
from os import path
import pkg_resources  # part of setuptools

__version__ = "0.1.0"
setup(
    name='dataScience Feature Selection Research',
    version=__version__,
    packages=find_packages(),

    url='https://github.com/starwalker/dataScience-featureselectionresearch',
    license='GNU General Public License v3.0',
    description='a package of feature Selection methods',
    zip_safe=True
)
