import sys
import setuptools
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text()

if sys.version_info < (3, 6):
    sys.exit('Python>=3.6 is required.')

setuptools.setup(
    name="petuum_medical_images",
    version="0.0.1a1",
    url="https://github.com/petuum/medical_images",

    description="The Petuum Medical Images Library built from "
                "Petuum Open Source projects.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='Apache License Version 2.0',
    packages=setuptools.find_packages(exclude=["scripts", "examples", "tests"]),
    include_package_data=True,
    platforms='any',

    install_requires=[
        'torch >= 1.5.0',
        'torchvision',
        'texar-pytorch >= 0.1.1',
        'typing>=3.7.4;python_version<"3.5"',
        'typing-inspect>=0.6.0',
        'forte',
        'nltk >= 3.5',
        'sklearn'
    ],
    extras_require={
    },
    entry_points={
    },
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)
