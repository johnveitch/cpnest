import os
import platform
import numpy
# Always prefer setuptools over distutils
from setuptools import setup
from setuptools import Extension

WINDOWS = platform.system().lower() == "windows"

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
    from Cython.Build import cythonize
    print('Running cython')
    ext_modules = [Extension("cpnest.parameter",
                             sources=[os.path.join("cpnest", "parameter.pyx")],
                             include_dirs=['cpnest', numpy.get_include()],
                             libraries=libraries,
                             extra_compile_args=["-O3","-ffast-math"])]
    ext_modules = cythonize(ext_modules)
else:
    # just compile the included parameter.c (already converted from
    # parameter.pyx) file
    ext_modules = [Extension("cpnest.parameter",
                             sources=[os.path.join("cpnest", "parameter.c")],
                             include_dirs=['cpnest', numpy.get_include()],
                             libraries=libraries,
                             extra_compile_args=["-O3","-ffast-math"])]

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the relevant file
with open(os.path.join(here, 'DESCRIPTION.rst'), encoding='utf-8') as f:
        long_description = f.read()


with open("requirements.txt") as requires_file:
    requirements = requires_file.read().split("\n")

setup(
        name='cpnest',
        use_scm_version=True,
        description='CPNest: Parallel nested sampling',
        long_description=long_description,
        author='Walter Del Pozzo, John Veitch',
        author_email='walter.delpozzo@ligo.org, john.veitch@ligo.org',
        url='https://github.com/johnveitch/cpnest',
        license='MIT',
        python_requires='>=3',
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
        install_requires=requirements,
        include_dirs = [numpy.get_include()],
        setup_requires=['numpy', 'cython', 'setuptools_scm'],
        package_data={"": ['*.c', '*.pyx', '*.pxd']},
        entry_points={},
        test_suite='tests',
        ext_modules=ext_modules
        )

