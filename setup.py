import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="drum-off",
    version="0.0.dev1",
    author="Mikko Hakila",
    author_email="hakila@gmail.com",
    description="Drum-Off, a live drumming game",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mikkoeinari/Drum-off",
    packages=['python'],
    python_requires='>=3.5, <4',
    install_requires=['pyaudio'],  # Optional
    package_data={
        'python': ['Sounds/*.wav','Sounds/bigkit/*.wav'],

    },

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GPLv3",
        "Operating System :: OS Independent",
    ],
)