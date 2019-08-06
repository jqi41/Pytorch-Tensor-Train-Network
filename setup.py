#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

try: 
    from setuptools import setup 
except ImportError:
    from distutils.core import setup 

with open('README.md') as readme_file:
    readme = readme_file.read()

#with open('HISTORY.rst') as history_file:
#    history = history_file.read()

setup(
    name = 'tc',
    version = '1.0',
    description = "Tensor-Train Network Library",
    long_description=readme + '\n\n',
    url='https://github.com/uwjunqi/Tensor-Train-Neural-Network/tree/master/tc',
    author="Jun Qi",
    author_email='jqi41@gatech.edu',
    packages=[
        'tc',
    ],
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
        'torch'
    ],
    license="LGPL",
    zip_safe=False,
    keywords='tc',
    classifiers=[
        'License :: OSI Approved :: ISC License (ISCL)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
 #   test_suite='tests',
 #   tests_require='pytest'
)
