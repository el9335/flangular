import os
from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()
    print required

os.system("easy_install pyStatParser")

setup(
		name ='flangular-nlp',
		version = '3.0',
		install_requires=required,
     )