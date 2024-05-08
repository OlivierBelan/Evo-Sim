from matplotlib.pyplot import annotate
from setuptools import setup
from Cython.Build import cythonize
import numpy
from distutils.extension import Extension
import pathlib
file_absolute_location:str = str(pathlib.Path(__file__).parent.resolve()) +"/"
file_running_location:str = ""
# file_absolute_location = ""
snn_cython = Extension(name='snn_cython',
        sources=[
            file_absolute_location + "SNN_cython/snn_cython.pyx",
            ],
        extra_compile_args=['-O3', '-march=native'],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
        )

record_cython = Extension(name='record_cython',
        sources=[
            file_absolute_location + "SNN_cython/record_cython.pyx"
            ],
        extra_compile_args=['-O3', '-march=native'],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
        )

runner_SL_cython = Extension(name='runner_SL_cython',
        sources=[
            file_absolute_location + "SNN_cython/runner_SL_cython.pyx"
            ],
        extra_compile_args=['-O3', '-march=native'],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
        )

runner_RL_cython = Extension(name='runner_RL_cython',
        sources=[
            file_absolute_location + "SNN_cython/runner_RL_cython.pyx"
            ],
        extra_compile_args=['-O3', '-march=native'],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
        )

tools_cython = Extension(name='tools_cython',
        sources=[
            file_absolute_location + "SNN_cython/tools_cython.pyx"
            ],
        extra_compile_args=['-O3', '-march=native'],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        )


setup(ext_modules=cythonize([snn_cython, runner_SL_cython, runner_RL_cython, record_cython, tools_cython],
    language_level=3, 
    annotate=False,
    # annotate=True, # generate html file with the cython code
    ),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
    # packages=['TOOLS_SNN']
)