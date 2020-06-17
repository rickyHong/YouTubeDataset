import os
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements.txt'
install_requires = [] # Examples: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()
    
    
setuptools.setup(
    name="YouTubeDataset", # Replace with your own username
    version="0.0.1",
    author="John Cumming",
    author_email="johnrcumming@gmail.com",
    description="Youtube Channel Dataset Downloader for PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/johnrcumming/YouTubeDataset",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=install_requires,
)