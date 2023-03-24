try:
    # pip >=20
    from pip._internal.network.session import PipSession
    from pip._internal.req import parse_requirements
except ImportError:
    try:
        # 10.0.0 <= pip <= 19.3.1
        from pip._internal.download import PipSession
        from pip._internal.req import parse_requirements
    except ImportError:
        # pip <= 9.0.3
        from pip.download import PipSession
        from pip.req import parse_requirements

requirements_file = "./requirements.txt"
install_reqs = parse_requirements(requirements_file, session=False)
dependencies = [str(ir.requirement) for ir in install_reqs]

from setuptools import setup, find_packages
import os
import sys
import subprocess

packages = [package for package in find_packages() if package.startswith('DeepFish')]

setup(name='DeepFish',
      version='1.0',
      description='DeepFish Instance Segmentation Code Base',
      author='Alzayat Saleh',
      author_email='alzayat.saleh@my.jcu.edu.au',
      license='Open-Source',
      url="https://github.com/alzayats/DeepFish",
      packages=packages,
      install_requires=dependencies,
)
