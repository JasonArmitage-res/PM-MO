import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PM_MO",
    version="1.0.0",
    author="Jason Armitage, Shramana Thakur , Rishi Tripathi",
    author_email="jason.armitage@uni-bonn.de, s6shthak@uni-bonn.de , s6ritrip@uni-bonn.de",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
