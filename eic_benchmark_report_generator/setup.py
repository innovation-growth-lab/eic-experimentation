from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
with open(f"requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read()

setup(
    name='eic_benchmark_report_generator',
    version='0.1.3',
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