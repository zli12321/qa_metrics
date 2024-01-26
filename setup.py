from setuptools import setup, find_packages
from setuptools.command.install import install
import os


setup(
    name='qa_metrics',
    version='0.1.24',
    author='Zongxia Li',
    author_email='zli12321@umd.edu',
    description='This package provides standard and classifier-based short form QA evaluation methods',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/zli12321/qa_metrics',
    packages=find_packages(),
    include_package_data=True,
    package_data={
    'qa_evaluators.metrics.classifier': ['*.pkl'],
    },
    install_requires=[
    'contractions>=0.0.1',
    'joblib',
    'requests',
    'scipy>=1.5.0',
    'scikit-learn==1.3.2',
    'numpy'
    ],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.6',

    )
