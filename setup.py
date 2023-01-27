import setuptools

with open("README.md", "r") as fh:

    long_description = fh.read()

setuptools.setup(
    name="ibge_gavb",  # Replace with your username
    version="1.0.0",
    author="GAVB",
    author_email="gavb@gmail.com",
    description="Collection of APIs for the IBGE Data Services in Brazil",
    long_description=long_description,
    long_description_content_type="Collection of APIs for the IBGE Data Services in Brazil",
    url="https://github.com/GAVB-SERVICOS/ibge_gavb",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.11.1",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
