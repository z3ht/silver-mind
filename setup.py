import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="silver-mind",
    version="1.0.0",
    author="Andrew Darling",
    author_email="andr3w.darling@gmail.com",
    description="Reinforcement Learning Chess AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/z3ht/silver-chess",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
