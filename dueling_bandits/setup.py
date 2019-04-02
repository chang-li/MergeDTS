#-*-coding: utf-8 -*-

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np


ext_modules = [Extension("_klucb", ["_klucb.pyx"]),
               Extension("_kllcb", ["_kllcb.pyx"]),
               ]

setup(
    cmdclass={'build_ext': build_ext},
    include_dirs=[np.get_include()],
    ext_modules=ext_modules
    )
