from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

DISTNAME = 'soft-dtw'
DESCRIPTION = "Python implementation of soft-DTW"
LONG_DESCRIPTION = open('README.rst').read()
MAINTAINER = 'Mathieu Blondel'
VERSION = '0.1.dev0'

# Cython 확장 모듈 설정
extensions = [
    Extension(
        name="sdtw.soft_dtw_fast",
        sources=["sdtw/soft_dtw_fast.pyx"],  # Cython 파일 경로
        include_dirs=[np.get_include()],    # NumPy 헤더 포함
    )
]

# setup 함수
setup(
    name=DISTNAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    maintainer=MAINTAINER,
    ext_modules=cythonize(extensions),  # Cython 컴파일
    zip_safe=False,  # .egg 파일에서 실행 금지
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: C',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
)
