# (c) WiPhy Development Team
# This library is released under the MIT License, see LICENSE.txt.

from codecs import open

from setuptools import setup
from wiphy.__init__ import WIPHY_VERSION

setup(name='wiphy',
      version=WIPHY_VERSION,
      description='The fundamental Python package for wireless signal processing at the physical layer',
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      url='https://github.com/ishikawalab/wiphy/',
      download_url="https://pypi.org/project/wiphy/",
      author='Ishikawa Laboratory',
      author_email='contact@ishikawa.cc',
      license='MIT',
      packages=['wiphy'],
      install_requires=[
          'numpy', 'pandas', 'scipy', 'sympy', 'numba', 'tqdm'
      ],
      classifiers=[
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.7'
      ],
      test_suite='wiphy.tests',
      zip_safe=False,
      package_data={'wiphy': ['util/inds/M=2_K=1_Q=2_minh=2_ineq=0.txt', 'util/inds/M=4_K=1_Q=4_minh=2_ineq=0.txt',
                              'util/inds/M=4_K=2_Q=4_minh=2_ineq=0.txt', 'util/inds/M=4_K=3_Q=4_minh=2_ineq=0.txt']},
      )
