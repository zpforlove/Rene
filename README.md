# Architecture #
![image](https://github.com/zpforlove/Rene/blob/main/images/Rene_model.png)
# Introduction #
This is a real-time respiratory disease recognition system based on the Rene (S) architecture, which can output the prediction probabilities of 15 diseases based on respiratory sounds. The selection of the confidence level Alpha for multimodal fusion is typically empirical and varies according to application scenarios and input patterns. Below is the performance of Rene model in the ICBHI respiratory disease monitoring task for reference only.


![image](https://github.com/zpforlove/Rene/blob/main/images/ICBHI.jpg)
# Prerequisites #
We used Python 3.9.7 and [PyTorch](https://pytorch.org/) 1.9.1+cu111 to train and test our model, but the codebase is expected to be compatible with Python 3.8-3.11 and the most recent version of PyTorch. The codebase also relies on some Python packages, the most notable dependencies are the foundation and backbone used in the construction of the Rene model: the general speech recognition model Whisper and the convolution-augmented Transformer for speech recognition: Conformer. The installation methods for these two dependencies can be found in the following two links.

[Whisper: Robust Speech Recognition via Large-Scale Weak Supervision
](https://github.com/openai/whisper)

[Conformer: Convolution-augmented Transformer for Speech Recognition](https://github.com/sooftware/conformer)

We also used the PyAudio library for real-time audio recording. PyAudio is the Python version of PortAudio, a cross-platform audio I/O library. The download method for PyAudio can be referred to the following link:

[PyAudio Download Website](https://pypi.org/project/PyAudio/) 



- Note for Debian / Ubuntu users: Please be sure to install the portaudio library development package (portaudio19-dev) and the python development package (python-all-dev) in advance. Please Follow these commands:

`sudo apt-get install python-all-dev`

`sudo apt-get install portaudio19-dev`

`pip install pyaudio`


- Note for Windows users: If the installation fails, please install PyAudio by the WHL file corresponding to your Python version from the following link : [PyAudio WHL Files](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio). You can execute a command similar to the one below in your console to install PyAudio.

`pip install PyAudio-0.2.11-cp37-cp37m-win_amd64.whl`

# Model #

**Rene.pth** can be downloaded through the following Google Drive share link. The purpose of open sourcing this model file is to encourage users to deploy the model on wearable respiratory sound detection devices, thereby promoting the development of edge artificial intelligence in the diagnosis of respiratory diseases.

[Rene.pth Download Path (Google Drive)](https://drive.google.com/file/d/1hGXNONeENRiom03RoXFwE1xWgyYzDlKK/view)

If you are a user from China, you can also download the model file through the following Baidu Netdisk share link:

[Rene.pth Download Path (Baidu Netdisk)](https://pan.baidu.com/s/1cqukxk2TQwJlyrez6ARroQ?pwd=c4iz)

# Usage #
Firstly, you need to ensure that the running device contains a sound card and corresponding drivers. For example, it cannot be a virtual machine or a server.

Connect the microphone to the stethoscope, then execute the following code to see the respiratory disease prediction probability values output in percentage form on the console.

`python ./streaming.py`

The dual-thread lung sound real-time decoding process is shown in the following figure.

![image](https://github.com/zpforlove/Rene/blob/main/images/Dual_threading.png)


# Author #

- Pengfei Zhang (PhD student in Bioscience and Biomedical Engineering at HKUST(GZ))
- Contacts: austin.zh@foxmail.com

 I appreciate any kind of feedback or contribution. Please feel free to contact me.

***Thou great star! What would be thy happiness if thou hadst not those for whom thou shinest!***

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;***--- Friedrich Wilhelm Nietzsche***



