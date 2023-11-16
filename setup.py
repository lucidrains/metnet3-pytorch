from setuptools import setup, find_packages

setup(
  name = 'metnet3-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.12',
  license='MIT',
  description = 'MetNet 3 - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/metnet3-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'vision transformers',
    'unet',
    'weather forecasting'
  ],
  install_requires=[
    'beartype',
    'einops>=0.7.0',
    'torch>=2.0',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
