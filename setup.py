import os
import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

source_pattern = './%s.pyx'

os.environ['CC'] = 'gcc-5'
os.environ['CXX'] = 'gcc-5'

kmeans_ext = [
    Extension('shieh.shieh_kmeans', sources=[source_pattern % "shieh_kmeans"],
              include_dirs=[numpy.get_include()],
              libraries=['m', 'z'],
              extra_compile_args = ['-O2', '-funroll-loops',],
              extra_link_args=['-lm',],
              )
    ]

setup(
    name = 'shieh.shieh_kmeans',
    ext_modules = cythonize(kmeans_ext),
    cmdclass = {'build_ext': build_ext},
)
