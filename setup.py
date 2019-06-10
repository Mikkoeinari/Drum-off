import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

required_libraries=['pyaudio>=0.2.11',
                    'scikit-learn>=0.2',
                    'sklearn>=0.0',
                    'scipy>=1.0',
                    'numpy>=1.11',
                    'pandas>=0.18',
                    'tensorflow>=1.3',
                    'keras>=2.2.4',
                    #'keras-tcn>=2.3.5',
                    'kivy>=1.1',
                    ]

package_data= ['Sounds/*.wav',
               'Sounds/bigkit/*.wav',
               'Sounds/bigkit/mono/*.wav',
               'UiImg/*.png']

setuptools.setup(
    name="drum_off",
    version="0.0.dev11",
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