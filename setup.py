from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = "pyafmrheo",
    version= "0.0.1",
    description=None,
    package_dir={'pyafmrheo': 'pyafmrheo'},
    install_requires=["numpy",
                      "pandas",
                      "scipy"
                      ],
    long_description=long_description,
    long_description_content_type = "text/markdown",
    packages=find_packages(),
    url="https://github.com/jlopezalo/pyafmrheo.git",
)