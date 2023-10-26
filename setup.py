from setuptools import setup, find_packages
from eic import PROJECT_DIR

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
with open(f"requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read()

setup(
    name='eic',
    version='0.0.0',
    license='MIT',
    description='',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='',
    packages=find_packages(
        exclude=["data"] #include tutorials folder to load utils from tutorials
    ),
    install_requires=[requirements],
    python_requires='>=3.9',
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
