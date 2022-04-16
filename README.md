# Basic-Chatbot
Chatbot baseline model

Build with Pytorch

# Description
A simple chatbot baseline model, trained on Transformer model and Persona-Chat dataset

Functionalities:
* Chat with you and make you life better!

# Usage
Developed on Python 3.8
1. Download required packages with
```
pip install -r requirements.txt
```
2. Train by running
```
python train.py
```
3. Trained models will be saved at /checkpoint
4. See the performence of each models by running
```
python test.py
```
5. Chat with chatbot by running (sentences are decoded with beam search)
```
python demo.py
```
