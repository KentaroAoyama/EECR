from setuptools import setup, find_packages

setup(
    name="eecr",
    version="0.0.1",
    description="Effective electrical conductivity simulator for rock",
    author="Kentaro Aoyama",
    packages=find_packages(),
    license="GPL-3.0",
    author_email="aoyama.kentaro.k0@elms.hokudai.ac.jp",
    url="https://github.com/KentaroAoyama/EECR",
    install_requires=["NumPy>=1.26.4", 
                      "SciPy>=1.13.1",
                      "iapws>=1.5.4",
                      ],
    package_data={"eecr": ["params/*.pkl"]}
)