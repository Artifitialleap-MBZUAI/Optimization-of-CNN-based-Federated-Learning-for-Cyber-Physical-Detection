# -*- coding: utf-8 -*-
"""


@author: Ammar.Abasi 2022
"""

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D, DepthwiseConv2D, SeparableConv2D 
from keras.layers import AvgPool2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras.constraints import maxnorm

import matplotlib.pyplot as plt
import numpy as np
import copy
import random
import sys
import pickle
import tensorflow
import keras
#import matplotlib.pyplot as plt
import numpy as np
import os
from keras.callbacks import LearningRateScheduler
from keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D
from keras.models import Input, Model
from keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
#from keras.utils.multi_gpu_utils import multi_gpu_model
from scipy.interpolate import splev, splrep
import pandas as pd


class Model():

    def __init__(self, loss, optimizer, classes=2,filters1=32,activation1='relu',kernel_size1=5,activation2='relu',kernel_size2=5,activation3='relu'):
        self.loss = loss
        self.optimizer = optimizer
        self.num_classes = classes
        self.filters1=filters1
        self.activation1 = activation1
        self.kernel_size1 = kernel_size1
        self.activation2 = activation2
        self.kernel_size2 = kernel_size2
        self.activation3 = activation3

    def fl_paper_model(self, train_shape):
        model = Sequential()
        
        # 1
        model.add(Conv1D(
            filters=self.filters1,
            kernel_size=self.kernel_size1,
            padding='valid',
            activation=self.activation1,#opt
            input_shape=train_shape,
            kernel_regularizer='l2',
        ))
        model.add(MaxPooling1D(
            pool_size=3,
            padding='same'
        ))
        model.add(Conv1D(
            filters=64,
            kernel_size=self.kernel_size2,
            padding='valid',
            activation=self.activation2,#opt
            kernel_regularizer='l2',
        ))
        model.add(MaxPooling1D(
            pool_size=3,
            padding='same'
        ))
        model.add(Dropout(0.8))

        # 2
     

        # 3
        model.add(Flatten())
        model.add(Dense(
            units=32,
            activation=self.activation3,
            kernel_regularizer='l2',
        ))
        
        
        # 4
        model.add(Dense(
            units=self.num_classes,
            activation='softmax'
        ))

        model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=['accuracy']
        )

        return model