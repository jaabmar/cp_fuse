from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cp_fuse",
    version="0.1.0",
    author="Javier Abad & Konstantin Donhauser",
    author_email="javier.abadmartinez@ai.ethz.ch",
    description="Python implementation of the methods introduced in the paper: Copyright-Protected Language Generation via Adaptive Model Fusion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jaabmar/cp-fuse",
    package_dir={"": "cp_fuse"},
    packages=find_packages(where="cp_fuse"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="machine learning, language models, copyright, model fusion",
    python_requires=">=3.12.3",
    install_requires=[
        "torch==2.3.0",
        "peft==0.10.0",
        "transformers==4.40.0",
        "datasets==2.19.1",
        "pynvml==11.5.0",
        "numpy==1.26.4",
        "pandas==2.2.2",
        "accelerate==0.30.1",
        "huggingface-hub==0.23.4",
        "bitsandbytes==0.44.1",
    ],
    entry_points={
        "console_scripts": [
            "train=examples.train:main",
            "evaluate=examples.evaluate:main",
        ],
    },
)
