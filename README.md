# graduation-design
This is the undergraduate graduation design of gj in ZJU, 2021. Title: Vision-based Fall Detection and System Design for the Elderly.



### Repository Structure

- singlenet-fucntion：train model and convert into *.tflite
  - |- dataset: preprocess dataset and data augmentation
  - |- estimation: some function used to do postprocess for inference outputs to show results
  - |-logs_singlenet：store logs -> loss + profiler + image
  - |- model: store models
  - |-model_struct: model structure, only use singlenet_func.py
  - |-tools: convert tensorflow model into tflite format
- singlenet-tflite-class：main project

- test4: simple android app to receive info

### Environment

#### Model

Packages requirement:

- Tensorflow == 2.5.0；Tensorboard == 2.5.0
- Tensorpack == 0.9.8
- numpy == 1.19.5
- opencv-contrib-python ==4.5.1.48
- opencv-python-headless == 3.4.8.29
- scipy
- pydot
- matplotlib
- Cython

Dataset:

- Microsoft COCO 2017

Train:

- Tesla V100 32GB: > 10h

#### Development Board - Rock Pi 4B

OS version:

- Ubuntu 20.04 LTS Server

Necessary Lib (need compilation manually):

- Tensorflow Lite == 2.5.0
- Opencv == 4.5.2 with **Opencv-Contrib-Module**
- Glog
- paho.mqtt.c & paho.mqtt.cpp

Optional Pip Package:

- Tensorflow-2.4.0-cp37-none-linux_aarch64
  - need to set global variable about architecture (ARM v8) for OpenBLAS

#### Remote Server

Only need to install **mosquitto**.



### Run

- adjust file directory
- build and run
  - only output key logs with **release** mode