try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

long_description = """\
This is a simple implementation of recommendation system using Collaborative Filtering algorithm.
Since this's built only on python modules, it's easy to see and understand how collaborative filtering algorithm works
"""

setup(
    name='pyrecmd',
    version='0.1.0',
    description=('pure python implementation of collaborative filtering for recommendation system'),
    long_description=long_description,
    py_modules=['pyrecmd'],
    install_requires=['numpy'],
    author='Humble Data Miner',
    author_email='humble.data.miner@gmail.com',
    url='https://github.com/humble-data-miner/pyrecmd',
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT license",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Operating System :: OS Independent"
    ]
)
