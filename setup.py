import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='dnnutil',
    version='0.0.2',
    author='Connor Anderson',
    author_email='connor.anderson@byu.edu',
    description='Utilities for working with deep neural networks in PyTorch',
    long_description=long_description,
    packages=setuptools.find_packages(),
    classifiers=(
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Unix',
    ),
)
