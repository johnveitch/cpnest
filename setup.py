#Always prefer setuptools over distutils
from setuptools import setup, find_packages
from setuptools import Extension
# To use a consistent encoding
from codecs import open
from os import path
import numpy

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'DESCRIPTION.rst'), encoding='utf-8') as f:
        long_description = f.read()

setup(
        name = 'cpnest',
        version = '0.1.1',
        description = 'CPNest: Parallel nested sampling',
        long_description=long_description,
        author = 'Walter Del Pozzo, John Veitch',
        author_email='walter.delpozzo@ligo.org, john.veitch@ligo.org',
        url='https://github.com/johnveitch/cpnest',
        license='MIT',
        classifiers =[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Data Analysts',
        'Topic :: Data Analysis :: Bayesian Inference',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        ],

        keywords='nested sampling bayesian inference',
        #packages=find_packages(exclude=['contrib','docs','tests*','examples']),
        packages=['cpnest'],
        install_requires=['numpy','scipy','cython'],
        setup_requires=['numpy'],
        # Don't know what this does
        extras_require={
            'dev': ['check-manifest'],
            'test': ['coverage'],
            },
        # Dictionary for data files
        # {'sample':['sample_file.dat']}
        package_data={"":['README.rst','LICENSE']},
        include_package_data=True,
        # To provide executable scripts, use entry points in preference to the
        # "scripts" keyword. Entry points provide cross-platform support and allow
        # pip to create the appropriate form of executable for the target platform.
        entry_points={
        #    'console_scripts':['sample=sample:main',
        #        ],
            },
            test_suite='tests',
        ext_modules=[
                Extension('cpnest.parameter',sources=['cpnest/parameter.pyx'],libraries=['m'],include_dirs=[numpy.get_include(),'cpnest/']),
                ]
        )

