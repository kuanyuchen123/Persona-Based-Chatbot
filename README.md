# Persona Based Chatbot
Trained on Transformer and Persona-Chat dataset

# Usage
Developed on Python 3.8 with Pytorch
1. Download required packages with
```
pip install -r requirements.txt
```
2. Train by running (trained models will be saved at /checkpoint)
```
python train.py
```
3. See the performence of each models by running
```
python test.py
```
4. Chat with chatbot by running (sentences are decoded with beam search)
```
python demo.py
```
