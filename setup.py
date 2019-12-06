from distutils.core import setup
import setuptools

with open("readme.md", "r") as fh:
    long_description = fh.read()

setup(
    name='max_act',
    version='0.0.1',
    author="Chandan Singh",
    description="Maximal activation in pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/csinva/max-activation-interpretation-pytorch",
    packages=setuptools.find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)