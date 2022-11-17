# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 12:02:57 2022

@author: Ammar.Abasi
"""

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from keras.datasets import cifar10
from keras.datasets import mnist
#from keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from keras.backend import image_data_format
from tensorflow.keras.optimizers import SGD


import matplotlib.pyplot as plt
import numpy as np
import copy
import random
import sys
import pickle
import tensorflow
import keras
#import matplotlib.pyplot as plt
#import numpy as np
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

base_dir = "dataset"

ir = 3 # interpolate interval
before = 2
after = 2
# normalize
scaler = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr))



from build_model2 import Model
#from hba_a  import HBA
import csv
import math
from numpy import linalg as LA
# client config
NUMOFCLIENTS = 10 # number of client(as particles) 10
SELECT_CLIENTS = 1 # c
EPOCHS = 1 # number of total iteration 30
CLIENT_EPOCHS = 50 # number of each client's iteration
BATCH_SIZE = 10 # Size of batches to train on
DROP_RATE = 0

# model config 
LOSS = 'sparse_categorical_crossentropy' # Loss function
NUMOFCLASSES = 2 # Number of classes
lr = 0.0025
# OPTIMIZER = SGD(lr=0.015, decay=0.01, nesterov=False)
OPTIMIZER = SGD(lr=lr, momentum=0.9, decay=lr/(EPOCHS*CLIENT_EPOCHS), nesterov=False) # lr = 0.015, 67 ~ 69%


def write_csv(method_name, list):
    file_name = '{name}_Apnea_HBA_randomDrop_{drop}%_output_C_{c}_LR_{lr}_CLI_{cli}_CLI_EPOCHS_{cli_epoch}_TOTAL_EPOCHS_{epochs}_BATCH_{batch}.csv'
    file_name = file_name.format(folder="origin_drop",drop=DROP_RATE, name=method_name, c=SELECT_CLIENTS, lr=lr, cli=NUMOFCLIENTS, cli_epoch=CLIENT_EPOCHS, epochs=EPOCHS, batch=BATCH_SIZE)
    f = open(file_name, 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    
    for l in list:
        wr.writerow(l)
    f.close()


def load_dataset():
    tm = np.arange(0, (before + 1 + after) * 60, step=1 / float(ir))

    with open(os.path.join(base_dir, "apnea-ecg.pkl"), 'rb') as f: # read preprocessing result
        apnea_ecg = pickle.load(f)

    X_train = []
    o_train, Y_train = apnea_ecg["o_train"], apnea_ecg["y_train"]
    groups_train = apnea_ecg["groups_train"]
    for i in range(len(o_train)):
        (rri_tm, rri_signal), (ampl_tm, ampl_siganl) = o_train[i]
		# Curve interpolation
        rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1) 
        ampl_interp_signal = splev(tm, splrep(ampl_tm, scaler(ampl_siganl), k=3), ext=1)
        X_train.append([rri_interp_signal, ampl_interp_signal])
    X_train = np.array(X_train, dtype="float32").transpose((0, 2, 1)) # convert to numpy format
    Y_train = np.array(Y_train, dtype="float32")

    X_test = []
    o_test, Y_test = apnea_ecg["o_test"], apnea_ecg["y_test"]
    groups_test = apnea_ecg["groups_test"]
    for i in range(len(o_test)):
        (rri_tm, rri_signal), (ampl_tm, ampl_siganl) = o_test[i]
		# Curve interpolation
        rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1)
        ampl_interp_signal = splev(tm, splrep(ampl_tm, scaler(ampl_siganl), k=3), ext=1)
        X_test.append([rri_interp_signal, ampl_interp_signal])
    X_test = np.array(X_test, dtype="float32").transpose((0, 2, 1))
    Y_test = np.array(Y_test, dtype="float32")

    #return X_train, Y_train, groups_train, X_test, Y_test, groups_test
    return (X_train, Y_train), (X_test, Y_test)



def init_model(train_data_shape):
    #model = Model(loss=LOSS, optimizer=OPTIMIZER, classes=NUMOFCLASSES)
    model =Model(loss=LOSS, optimizer=OPTIMIZER, classes=NUMOFCLASSES,filters1=32,activation1='relu',kernel_size1=5,activation2='relu',kernel_size2=5,activation3='relu')
    fl_model = model.fl_paper_model(train_shape=train_data_shape)

    return fl_model



def client_data_config(X_train, Y_train):
    client_data = [() for _ in range(NUMOFCLIENTS)] # () for _ in range(NUMOFCLIENTS)
    num_of_each_dataset = int(X_train.shape[0] / NUMOFCLIENTS)
    
    for i in range(NUMOFCLIENTS):
        split_data_index = []
        while len(split_data_index) < num_of_each_dataset:
            item = random.choice(range(X_train.shape[0]))
            if item not in split_data_index:
                split_data_index.append(item)
        
        new_X_train = np.asarray([X_train[k] for k in split_data_index])
        new_Y_train = np.asarray([Y_train[k] for k in split_data_index])
    
        client_data[i] = (new_X_train, new_Y_train)

    return client_data

 
def fedAVG(server_weight):
    avg_weight = np.array(server_weight[0]) #matrix
    #avg_weight = np.reshape(avg_weight,np.shape(server_weight))
    
    if len(server_weight) > 1: 
        for i in range(1, len(server_weight)):
            avg_weight += server_weight[i]
    
    avg_weight = avg_weight / len(server_weight)

    return avg_weight

 
def client_update(index, client, now_epoch, avg_weight,train_data_shape):
    print("client {}/{} fitting".format(index + 1, int(NUMOFCLIENTS * SELECT_CLIENTS)))

    if now_epoch != 0:
        client.set_weights(avg_weight) 
        
    pop = 10                    # Honey Badger population size.50
    MaxIter = 100               # Maximum number of iterations.200
    dim = 5                    # The dimension.
    #fl=[-10,-9,10]                   # The lower bound of the search interval.
    #ul=[-10,-9,30]                       # The upper bound of the search interval.
    # activation1, kernel_size1,activation2, kernel_size2,activation3
    lb =[0,0,0,0,0]#fl*np.ones([dim, 1])
    ub =[2,2,2,2,2]  #ul*np.ones([dim, 1])
  
    client, GbestScore, GbestPositon, Curve = hba(index,pop,MaxIter,dim,lb,ub,client_data,train_data_shape,CLIENT_EPOCHS)
    
    #client.fit(client_data[index][0], client_data[index][1],
        #epochs=CLIENT_EPOCHS,
        #batch_size=BATCH_SIZE,
       # verbose=1,
        #validation_split=0.2,
        
    #)

    return client,GbestPositon


def hba(index,pop,Max_iter,dim,lb,ub,client_data,train_data_shape,CLIENT_EPOCHS):
        
        X = initial( pop,  dim,  lb, ub)                    # Initialize the number of honey badgers
        fitness = np.zeros([pop, 1])       
        for i in range(pop):
          fitness[i] = CaculateFitness1(X[i, :], client_data,train_data_shape,index,CLIENT_EPOCHS)
        fitness, sortIndex = SortFitness(fitness)       # Sort the fitness values of honey badger.
        X = SortPosition(X, sortIndex)                  # Sort the honey badger.
        GbestScore = fitness[0]                         # The optimal value for the current iteration.
        GbestPositon = np.zeros([1, dim])
        GbestPositon[0, :] = X[0, :]
        Curve = np.zeros([Max_iter, 1])
        C = 2                                          # constant in Eq. (3)
        beta = 6                                       # the ability of HB to get the food  Eq.(4)
        vec_flag=[1,-1]
        vec_flag=np.array(vec_flag)
        Xnew = np.zeros([pop, dim])
        for t in range(Max_iter):
            print("iteration: ",t)
            alpha=C*math.exp(-t/Max_iter);             # density factor in Eq. (3)
            #print(alpha)
           
            I=Intensity(pop,GbestPositon,X);           # intensity in Eq. (2)
            Vs=random.random()
            for i in range(pop):
              Vs=random.random()
              F=vec_flag[math.floor((2*random.random()))]
              for j in range(dim):
                di=GbestPositon[0,j]-X[i,j]
                if (Vs <0.5):                           # Digging phase Eq. (4)
                  r3=np.random.random()
                  r4=np.random.randn()
                  r5=np.random.randn()
                  Xnew[i,j]=GbestPositon[0,j] +F*beta*I[i]* GbestPositon[0,j]+F*r3*alpha*(di)*np.abs(math.cos(2*math.pi*r4)*(1-math.cos(2*math.pi*r5)));
                else:
                  r7=random.random()
                  Xnew[i,j]=GbestPositon[0,j]+F*r7*alpha*di;    # Honey phase Eq. (6)
              
              
              Xnew[i,:] = BorderCheck1(Xnew[i,:], lb, ub, dim)
              #print(Xnew[i,:])
              Xnew[i,:][np.isnan(Xnew[i,:])] = 0
              Xnew[i,:]=[round(item) for item in Xnew[i,:]]
              #print(Xnew[i,:])
              tempFitness = CaculateFitness1(X[i, :], client_data,train_data_shape,index,CLIENT_EPOCHS)
              if (tempFitness >= fitness[i]):
                fitness[i] = tempFitness               
                X[i,:] = Xnew[i,:] 
            for i in range(pop):                         
              X[i,:] = BorderCheck1(X[i,:], lb, ub ,dim)
            Ybest,index01 = SortFitness(fitness)               # Sort fitness values.
            if (Ybest[0] >= GbestScore):                          
              GbestScore = Ybest[0]     # Update the global optimal solution.
              GbestPositon[0, :] = X[index01[0], :]           # Sort fitness values 
            Curve[t] = GbestScore
            
        
        GbestPositon[0,:]=[round(item) for item in GbestPositon[0,:]]
        print(GbestPositon[0,:])
        LOSS = 'sparse_categorical_crossentropy' 
        lr = 0.0025
        #EPOCHS = 2 # number of total iteration 30
        #CLIENT_EPOCHS = 5 # number of each client's iteration
        OPTIMIZER = SGD(lr=lr, momentum=0.9, decay=lr/(EPOCHS*CLIENT_EPOCHS), nesterov=False) # lr = 0.015, 67 ~ 69%
        #NUMOFCLASSES = 2
        activation1=GbestPositon[0][0]
        kernel_size1=GbestPositon[0][1]
        activation2=GbestPositon[0][2]
        kernel_size2=GbestPositon[0][3]
        activation3=GbestPositon[0][4]    
        activation1=GetActivationbyValue(activation1)
        kernel_size1=GetAkernelSizebyValue(kernel_size1)
        activation2=GetActivationbyValue(activation2)
        kernel_size2=GetAkernelSizebyValue(kernel_size2)
        activation3=GetActivationbyValue(activation3) 
        
        
        model2 = Model(loss=LOSS, optimizer=OPTIMIZER, classes=NUMOFCLASSES,filters1=32,activation1=activation1,kernel_size1=5,activation2=activation2,kernel_size2=5,activation3=activation3)
       
        client = model2.fl_paper_model(train_shape=train_data_shape)
        #model = Model(loss=LOSS, optimizer=OPTIMIZER, classes=NUMOFCLASSES)
        #client = model.fl_paper_model(train_shape=train_data_shape)
        client.fit(client_data[index][0], client_data[index][1],
            epochs=CLIENT_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1,
            validation_split=0.2,
        )
        return client,GbestScore,GbestPositon,Curve
    
    #rng = np.random.default_rng()
    #time_start = time.time()
    
    #GbestScore, GbestPositon, Curve = hba(client_data)
    # This function is to initialize the Honey Badger population.
def initial(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = random.random()*(ub[j] - lb[j]) + lb[j]
            #print(X[i, j])
    return X

def GetActivationbyValue(value):        
    if (value==0):
       value='sigmoid'
    elif (value==1):
        value='relu'
    elif (value==2):
        value='tanh'
    else:
        value='relu'    

    return value
def GetAkernelSizebyValue(value):        
    if (value==0):
       value=3
    elif (value==1):
        value=5
    elif (value==2):
        value=7
    else:
        value=5    

    return value

# Calculate fitness values for each Honey Badger.
def CaculateFitness1(X,client_data,train_data_shape,ind,CLIENT_EPOCHS):
    LOSS = 'sparse_categorical_crossentropy' 
    lr = 0.0025
    EPOCHS = 2 # number of total iteration 30
    
    OPTIMIZER = SGD(lr=lr, momentum=0.9, decay=lr/(EPOCHS*CLIENT_EPOCHS), nesterov=False) # lr = 0.015, 67 ~ 69%
    NUMOFCLASSES = 2     
    activation1=X[0]
    kernel_size1=X[1]
    activation2=X[2]
    kernel_size2=X[3]
    activation3=X[4]    
    activation1=GetActivationbyValue(activation1)
    kernel_size1=GetAkernelSizebyValue(kernel_size1)
    activation2=GetActivationbyValue(activation2)
    kernel_size2=GetAkernelSizebyValue(kernel_size2)
    activation3=GetActivationbyValue(activation3)    
    
    model = Model(loss=LOSS, optimizer=OPTIMIZER, classes=NUMOFCLASSES,filters1=32,activation1=activation1,kernel_size1=kernel_size1,activation2=activation2,kernel_size2=kernel_size2,activation3=activation3)
    client = model.fl_paper_model(train_shape=train_data_shape)
    history = client.fit(client_data[ind][0], client_data[ind][1],
    epochs=CLIENT_EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_split=0.2,
    
       

    )
    
    #print("1")
    #print(history.history.keys())
    #print("2")
    #print(history.history['val_accuracy'])
    #print("3")
    #print(history.history['accuracy'])
    #print("4")
    #print(history.history['accuracy'][-1])
    #fitness= client['accuracy']
    fitness=history.history['accuracy'][-1]
    return fitness

    

  
# Sort fitness.
def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index01 = np.argsort(Fit, axis=0)
    return fitness,index01


# Sort the position of the Honey Badger according to fitness.
def SortPosition(X,index01):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i,:] = X[index01[i],:]
    return Xnew


# Boundary detection function.
def BorderCheck1(X,lb,ub,dim):
        for j in range(dim):
            if X[j]<lb[j]:
                X[j] = ub[j]
            elif X[j]>ub[j]:
                X[j] = lb[j]
        return X
def Intensity(pop,GbestPositon,X):
  epsilon = 0.00000000000000022204
  di = np.zeros(pop)
  S = np.zeros(pop)
  I = np.zeros(pop)
  for j in range(pop):
    if (j <= pop):
      di[j]=LA.norm([[X[j,:]-GbestPositon+epsilon]])
      S[j]= LA.norm([X[j,:]-X[j+1,:]+epsilon])
      di[j] = np.power(di[j], 2)
      S[j]= np.power(S[j], 2)
    else:
      di[j]=[ LA.norm[[X[pop,:]-GbestPositon+epsilon]]]
      S[j]=[LA.norm[[X[pop,:]-X[1,:]+epsilon]]]
      di[j] = np.power(di[j], 2)
      S[j]= np.power(S[j], 2)    
  
    for i in range(pop):
      n = random.random()
      I[i] = n*S[i]/[4*math.pi*di[i]]
    return I
   
    


if __name__ == "__main__":
    (X_train, Y_train), (X_test, Y_test) = load_dataset()

    server_model = init_model(train_data_shape=X_train.shape[1:])
    server_model.summary()

    client_data = client_data_config(X_train, Y_train)
    fl_model = []
    
    for i in range(NUMOFCLIENTS):
        
        fl_model.append(init_model(train_data_shape=client_data[i][0].shape[1:]))

    avg_weight = server_model.get_weights()
    server_evaluate_acc = []

    for epoch in range(EPOCHS):  
        server_weight = []
        
        
        
        
        selected_num = int(max(NUMOFCLIENTS * SELECT_CLIENTS, 1))
        split_data_index = []
        while len(split_data_index) < selected_num:
            item = random.choice(range(len(fl_model)))
            if item not in split_data_index:
                split_data_index.append(item)
        split_data_index.sort()
        selected_model = [fl_model[k] for k in split_data_index]
        
        
        

        for index, client in enumerate(selected_model):
            recv_model,GbestPositon = client_update(index, client, epoch, avg_weight,train_data_shape=X_train.shape[1:])
            
            
            
            rand = random.randint(0,99)
            drop_communication = range(DROP_RATE)
            if rand not in drop_communication:
                server_weight.append(copy.deepcopy(recv_model.get_weights()))
                 
        avg_weight = fedAVG(server_weight)

        server_model.set_weights(avg_weight)
        print("server {}/{} evaluate".format(epoch + 1, EPOCHS))
        server_evaluate_acc.append(server_model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE, verbose=1))

    write_csv("FedAvg", server_evaluate_acc)

