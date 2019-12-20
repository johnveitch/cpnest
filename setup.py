# Always prefer setuptools over distutils
import numpy
from setuptools import setup, find_packages
from setuptools import Extension
from setuptools.command.build_ext import build_ext as _build_ext
# To use a consistent encoding
from codecs import open
import os
import re
import platform

WINDOWS = platform.system().lower() == "windows"


# see https://stackoverflow.com/a/21621689/1862861 for why this is here
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        self.include_dirs.append(numpy.get_include())


# check whether user has Cython
try:
    import Cython
except ImportError:
    have_cython = False
else:
    have_cython = True

# set extension
libraries = [] if WINDOWS else ["m"]
if have_cython:  # convert the pyx file to a .c file if cython is available
    ext_modules = [Extension("cpnest.parameter",
                             sources=[os.path.join("cpnest", "parameter.pyx")],
                             include_dirs=['cpnest'],
                             libraries=libraries)]
else:
    # just compile the included parameter.c (already converted from
    # parameter.pyx) file
    ext_modules = [Extension("cpnest.parameter",
                             sources=[os.path.join("cpnest", "parameter.c")],
                             include_dirs=['cpnest'],
                             libraries=libraries)]

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
        name='cpnest',
        version=VERSION,
        description='CPNest: Parallel nested sampling',
        long_description=long_description,
        author='Walter Del Pozzo, John Veitch',
        author_email='walter.delpozzo@ligo.org, john.veitch@ligo.org',
        url='https://github.com/johnveitch/cpnest',
        license='MIT',
        cmdclass={'build_ext': build_ext},
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.2',
            'Programming Language :: Python :: 3.3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8'],
        keywords='nested sampling bayesian inference',
        packages=['cpnest'],
        install_requires=['numpy', 'scipy', 'corner', 'tqdm', 'cython'],

        setup_requires=['numpy', 'scipy', 'cython'],
        tests_require=['corner','tqdm'],
        package_data={"": ['*.c', '*.pyx', '*.pxd']},
        entry_points={},
        test_suite='tests',
        ext_modules=ext_modules
        )

