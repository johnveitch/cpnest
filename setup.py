#Always prefer setuptools over distutils
from setuptools import setup, find_packages
from setuptools import Extension
from setuptools.command.build_ext import build_ext as _build_ext
# To use a consistent encoding
from codecs import open
import os
import re

# see https://stackoverflow.com/a/21621689/1862861 for why this is here
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

# check whether user has Cython
try:
    import Cython
except ImportError:
    have_cython = False
else:
    have_cython = True

# set extension
if have_cython: # convert the pyx file to a .c file if cython is available
    ext_modules = [ Extension("cpnest.parameter", sources =[ "cpnest/parameter.pyx"], include_dirs=['cpnest/'], libraries=['m']) ]
else: # just compile the included parameter.c (already converted from parameter.pyx) file
    ext_modules = [ Extension("cpnest.parameter", sources =[ "cpnest/parameter.c"], include_dirs=['cpnest/'], libraries=['m']) ]

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the relevant file
with open(os.path.join(here, 'DESCRIPTION.rst'), encoding='utf-8') as f:
        long_description = f.read()

# Get the version info from __init__.py file
def readfile(filename):
    with open(filename) as fp:
        filecontents = fp.read()
    return filecontents

VERSION_REGEX = re.compile(r"__version__ = \'(.*?)\'")
CONTENTS = readfile(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "cpnest", "__init__.py"))
VERSION = VERSION_REGEX.findall(CONTENTS)[0]

setup(
        name = 'cpnest',
        version = VERSION,
        description = 'CPNest: Parallel nested sampling',
        long_description=long_description,
        author = 'Walter Del Pozzo, John Veitch',
        author_email='walter.delpozzo@ligo.org, john.veitch@ligo.org',
        url='https://github.com/johnveitch/cpnest',
        license='MIT',
        cmdclass = {'build_ext': build_ext},
        classifiers =[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        #'Intended Audience :: Developers',
        #'Topic :: Data Analysis :: Bayesian Inference',

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
        install_requires=['numpy','scipy','corner'],
        setup_requires=['numpy'],
        tests_require=['corner'],
        package_data={"": ['*.c', '*.pyx', '*.pxd']},
        # To provide executable scripts, use entry points in preference to the
        # "scripts" keyword. Entry points provide cross-platform support and allow
        # pip to create the appropriate form of executable for the target platform.
        entry_points={
        #    'console_scripts':['sample=sample:main',
        #        ],
            },
            test_suite='tests',
        ext_modules=ext_modules
        )

