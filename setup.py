from setuptools import setup, find_packages
    
# Version Number:
version = '0.0.1'

setup(
    name="dss",
    version=version,
    url="",
    description="Diffusion structure search",
    install_requires=[
        "numpy",
        "torch",
        "tensorboard",
        "schnetpack"
    ],
    python_requires=">=3.5",
    packages=find_packages(),
)
