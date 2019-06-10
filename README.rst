========
Drum-Off
========

Drum-Off is a game where a drummer and the computer take turns in playing a beat.
This is part of my thesis "Automatic Transcription and Quantization of Percussive Audio and Generation of New Sequences With a Live Drumming Game Application, Drum-Off"

Required Libraries
==================
- pyaudio: For audio IO
- sklearn: Clustering
- scipy: Signal processing
- numpy: array and matrix operations
- pandas: CSV file io
- tensorflow: Neural networks
- keras: Neural networks
- keras-tcn: TCN network
- kivy: User Interface

Installing
==========
The development is in progress and the code can be forked and run at own risk.

- Clone the repository
- Navigate to folder "drum_off"
- Run the game with command::

    python drumoff.py

Instructions for playing the Game
=================================

For playing this game you need a system capable of running the code, microphone to record the drumming and some way of playing pack the generated or transcribed parts.
The code has been tested with a macbook laptop computer it's the internal microphone and the speakers or headphones connected to the headphone jack.
The recording level needs to be adjusted so that as little as possible distortion is produces. With acoustic drums distortion will occur.
Using a made up drumkit with something like coffecups as hi-hats, table as a snare drum and the floor as a kick drum can be used if a drumkit is not available.

Start screen:
-------------

- If you are running the game for the first time or need to use a new drum kit select "Soundcheck new drumkit"
- If you have sounchecked a drumkit in a previoius session select "Load drumkit"
- If you have a drumkit loaded select "Drum-Off!"
- If you wish to quit select "Quit"

Soundcheck screen:
------------------

- Type a name for your drumkit
- Record 16 hits per drum by pressing the red record button button next to drum name, playing the hits and then pressing the blue stop button.
    - You may audit your samples by pressing the green play button after recording the samples. This helps in debugging performance problems
- Some drums have multiple sample options next to their control buttons. For hi-hat these resemble: close, open and hihat stomp hits. For tom-toms and cymbals the numbers represent different drums. Select the number and perform separate soundcheck per each drum/playing style.
- After recording all drum's samples press"finish soundcheck" to process the sample audio, prepare the learning model and return to Start screen.
    - Different neural network models can be selected from the "Model type" radio buttons, the default "Multi In-Multi Out" model is recommended.

Load Drumkit sceen:
-------------------

- Navigate to the folder containing your drumkit, select the folder and press "load". This will return you to the Start screen.

Drum-Off! screen:
-----------------

- Start playing by pressing "Play".

- If you have the option "Stop after perform" option selected you may play back your performance after pressing "stop" by selecting "Play Back Last Performance."
    - if you have "Computer adversary" enabled you can play the computer turn by pressing "computer" after stopping your performance.
    - If you have "Stop after perform" disabled, your performance or computer performance is played back automatically after your turn.
    - Playing with "Computer Adversary" enabled and "Stop after perform" is the intended game mode, the optional behaviour can be used for debugging.
- All other options are used for debugging and tuning performance
    - "Update Model" can be used to prevent saving the model after learning a new part.
    - "Use adjustment" can be used to change all drums onset detection threshold if source separation does not perform satisfactory.
        - more likely a new soundcheck is needed.
    - "use default 1.33" and the corresponding slider can be used to oversample the training data, bootstrapping up to 12 times the data. This will lead to overfitting.
        - Currently not relevant with "Multi In-Multi Out" model.
    - "dynamic temp" and "Temperature" slider can be used to introduce more or less variance to the generated sequences. The artistic freedom the computer exhibits.
    - Adjusting learning rate is currently offline, it is no longer relevant.
    - "Use Quantization" can be used to compare learning from quantized or original performance.



