import sys
from setuptools import setup, find_packages

setup(name = "palindrome",
      version = palindrome.version.__version__,
      packages = find_packages(),
      description = "Tools to calculate palindromes",
      author = "John Doe",
      author_email = "alexeys@princeton.edu",
      maintainer = "Jim Pivarski (DIANA-HEP)",
      maintainer_email = "pivarski@fnal.gov",
      download_url = "https://github.com/palindrome/palindrome-python",
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
