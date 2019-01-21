import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dfem",
    version="1.0.0",
    author="Mikhail",
    author_email="maksimau.mikhail@gmail.com",
    description="fem package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VonFrundsberg/FEM",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
