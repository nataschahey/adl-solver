from setuptools import setup, find_packages

setup(
    name='adl-solver',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
    ],
    description='Numerical solver for risk-based auto-deleveraging in cross-margin exchanges',
    author='Natascha Hey, Steven Campbell, Marcel Nutz, Ciamac Moallemi',
    python_requires='>=3.8',
)