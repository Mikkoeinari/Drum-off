import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

required_libraries=['pyaudio',
                    'scikit-learn',
                    'sklearn',
                    'scipy',
                    'numpy',
                    'pandas',
                    'tensorflow',
                    'keras',
                    #'keras-tcn>=2.3.5',
                    'kivy',
                    ]

package_data= ['Sounds/*.wav',
               'Sounds/bigkit/*.wav',
               'Sounds/bigkit/mono/*.wav',
               'UiImg/*.png',
               'UI.kv',
               'click.wav',
               'countIn.csv']

setuptools.setup(
    name="drum_off",
    version="0.0.dev19",
    author="Mikko Hakila",
    author_email="hakila@gmail.com",
    description="Drum-Off, a live drumming game",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mikkoeinari/Drum-off",
    packages=['drum_off'],
    python_requires='>=3.5, <4',
    install_requires=required_libraries,
    package_data={
        'drum_off':package_data,
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)