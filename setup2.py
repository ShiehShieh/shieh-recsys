import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from distutils.core import setup
from Cython.Build import cythonize


ext = [
    Extension('test', sources=['test.pyx'],
              include_dirs=[numpy.get_include()],
              libraries=['m', 'z'],
              extra_compile_args = ['-O2', '-funroll-loops',],
              extra_link_args=['-lm',],
              )
    ]


setup(
    name = 'test',
    ext_modules = cythonize(ext),
)
