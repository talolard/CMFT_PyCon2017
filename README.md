# Convolutional Methods for Text Workshop 
## Excersices for the workshop given at PyCon Israel 2017 

## Intro 
This repository has the excersises for the workshop on convolutional methods for text. You can read about the constructs we'll be using and why you migh want to consider convolutions in my [blog post](https://medium.com/@TalPerry/convolutional-methods-for-text-d5260fd5675f). In a nutshell, convolutions run in parrelell so they are much faster than RNNs. 

The core task in this repository is to restore missing punctuation and capitalisation in a text, something that we may find ourselves doing when working with speech recognition systems.
To avoid boilerplate overhead, we are doing this at the charecter level, not word. 
## Getting started
Clone this repository
```
pip3 install tensorflow nltk 
```

## Contents
`train.py` is your entry point for these tasks. To run a task modify the model import at the top
```python
from models.q1_lstm_baseline import LSTMBaseline
```
And then
```
python3 train.py
```

the *models* directory contains each of the "questions" as well as some boilerplate as follows

`base.py` Holds a base class that covers the boiler plate for a NN. Includes setting up the loss and optimiser as well as embedding charecters. You should be able to ignore this during the workshop
`densenet.ops` The densenet folder has all of the ops you may need to implement the various tasks here. I include it for reference but you won't get much from the workshop if you just copy paste it so try not to look here if you are stuck.
## Tasks
All tasks are described below. You are of course free to do whichever ones you like. They are arranged in order of difficulty and build on one another
### `q1_lstm_baseline`
In this task you'll use an LSTM or bidirectional LSTM to solve the task. Do this if you are new to Tensorflow/NN or you want to make your own baseline to see how slow LSTMs are compared to convolutions

### `q2_simple_conv` 
In this task we'll implement our own convolution op for 1d seqeunces like text. Once we have the op we'll apply it to our inputs to try and restore missing punctuation and capitalisation. 

### `q3_ResidualConvs` 
This task builds on the previous one. We'll want to increase our receptive field without encountering vanishing gradients. To do that we'll use residual connection a la DenseNet which you will implement. 

### `q4_DilatedConvs`
In this task we'll implement 1d dilated convolutions and use them as another method to increase receptive field without vanishing gradients. Bonus if you build residual dilated convolutions

### `q5_deconvolution_autoencoder`
In this question we'll use the residual convs we built and add 1d pooling and 1d "Deconvolutions" to encode our input as a vector and restore it respectively. 