import setuptools


VERSION = open('VERSION').read()
LONG_DESCRIPTION = open('README.md').read()

with open("requirements.txt", "r") as fp:
    reqs = fp.read().split("\n")

setuptools.setup(
    author="GAVB Servicos de Informatica LTDA",
    name="ibge",
    license="MIT",
    description="Collection of APIs for the IBGE Data Services in Brazil",
    version=VERSION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires=">=3.11",
    install_requires=reqs,
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Natural Language :: Portuguese (Brazilian)",
    ],
    setup_requires=["pytest_runner"],
    tests_require=["pytest== 7.2.1"],
    tests_suite="tests",
)
