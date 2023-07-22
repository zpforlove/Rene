![avatar](/image/Rene.png)
## Hello! This project is the real-time respiratory disease monitoring system designed based on the Rene model architecture. ##
# Introduction #
This is a dual-thread real-time respiratory disease discrimination model designed based on the Rene(S) architecture. It can integrate with the LightGBM model trained with comprehensive medical record according to the confidence of Alpha, to output the probability values of 15 types of diseases. The selection of the confidence level Alpha is typically empirical and varies according to application scenarios and input patterns. Below is the performance of Rene model in the ICBHI respiratory disease monitoring task for reference only.


![image](/image/ICBHI.jpg)
# Prerequisites #
We used Python 3.9.7 and PyTorch 1.9.1+cu111 to train and test our model, but the codebase is expected to be compatible with Python 3.8-3.11 and the most recent version of PyTorch. The codebase also relies on some Python packages, the most notable dependencies are two foundations used in the construction of the Rene model: the general speech recognition model Whisper and the convolution-augmented Transformer for speech recognition: Conformer. The installation methods for these two dependencies can be found in the following two links.

[Whisper: Robust Speech Recognition via Large-Scale Weak Supervision
](https://github.com/openai/whisper)

[Conformer: Convolution-augmented Transformer for Speech Recognition](https://github.com/sooftware/conformer)

We also used the PyAudio library for real-time audio recording. PyAudio is the Python version of PortAudio, a cross-platform audio I/O library. The download method for PyAudio can be referred to the following link:

[Pyaudio Download Website](https://pypi.org/project/PyAudio/) 



- Note for Debian / Ubuntu users: Please be sure to install the portaudio library development package (portaudio19-dev) and the python development package (python-all-dev) in advance. Please Follow these commands:

`sudo apt-get install python-all-dev`

`sudo apt-get install portaudio19-dev`

`pip install pyaudio`


- Note for Windows users: If the installation fails, please install by the PyAudio whl file corresponding to your Python version from the following link : [Pyaudio Whl Files](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio)

# Model #

We have released the model file **Rene.pth** in the Releases module, encouraging users to deploy the model on wearable respiratory sound detection devices, thus promoting the development of edge artificial intelligence in respiratory disease diagnosis.

# Usage #
Connect the microphone to the stethoscope, then execute the following code to see the respiratory disease prediction probability values output in percentage form on the console.

`python ./streaming.py`

Note: By default, the model is inputted with a 10-second audio clip collected by the microphone for disease determination each time, so there will be some delay in the disease prediction relative to the audio input.


# Author #

- Pengfei Zhang (PhD of Bioscience and Biomedical Engineering at HKUST(GZ))
- Contacts: austin.zh@foxmail.com

 I appreciate any kind of feedback or contribution. Please feel free to contact me.

***Thou great star! What would be thy happiness if thou hadst not those for whom thou shinest!***

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;***--- Friedrich Wilhelm Nietzsche***



