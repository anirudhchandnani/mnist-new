from keras.models import Sequential
from keras.layers import Dense
import numpy as np
#from scipy.misc import imread, imresize,imshow
import tensorflow as tf
from keras.models import load_model
from keras.models import model_from_json
import json


def init():

    '''model = Sequential()

    #our sequential type for the neural network model has been assigned to variable model

    #add the first input layer with activation of ReLU
    model.add(Dense(784, activation = 'relu'))

    #add first hidden layer with activation of ReLU
    model.add(Dense(1000, activation = 'relu'))

    #add second hidden layer with activation of ReLU
    model.add(Dense(800, activation = 'relu'))
    #model.add(Flatten())
    #add final output layer with activation - softmax
    model.add(Dense(10, activation = 'softmax'))

    #architechture remains same as that of the train.py script
'''
   # load json and create model
    model = load_model('final_model_cnn (1).h5')
    
    from keras.losses import categorical_crossentropy
    
    model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])

    #graph = tf.compat.v1.get_default_graph()
    #graph = tf.reset_default_graph()
    from tensorflow.python.framework import ops
    graph = ops.reset_default_graph()
    return model, graph
