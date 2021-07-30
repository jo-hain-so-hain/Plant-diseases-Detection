import tensorflow as tf

import pickle

import numpy as np

import keras

import os

done=False

model=0
labels=0

def loadmodel():
    model = pickle.load(open(os.getcwd()+"/predictor/cnn_model.pkl", 'rb'))
    labels = pickle.load(open(os.getcwd()+"/predictor/label_transform.pkl",'rb')).classes_
    done=True