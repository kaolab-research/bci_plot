import setuptools

print(setuptools.find_packages())
if __name__ == '__main__':
    setuptools.setup(
        name="bci_plot",
        version="0.0.1",
        author="NECL",
        author_email="",
        description="package for running paper code",
        packages=setuptools.find_packages(exclude=("tests")),
    )
