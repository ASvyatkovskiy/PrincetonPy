import sys
from setuptools import setup, find_packages

setup(name = "palindrome",
      version = "0.0.3",
      packages = find_packages(),
      description = "Tools to calculate palindromes",
      author = "John Doe",
      author_email = "alexeys@princeton.edu",
      download_url = "https://github.com/ASvyatkovskiy/PrincetonPy/tree/master/Session4",
      license = "Apache Software License v2",
      test_suite = "tests",
      install_requires = [],
      tests_require = [],
      classifiers = ["Development Status :: 5 - Production/Stable",
                     "Environment :: Console",
                     "Intended Audience :: Science/Research",
                     "License :: OSI Approved :: Apache Software License",
                     "Topic :: Scientific/Engineering :: Mathematics",
                     ],
      platforms = "Any",
      )
