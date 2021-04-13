from __future__ import absolute_import, division, print_function
from __future__ import print_function
# -*- coding: utf-8 -*-
"""
Spyder Editor

Radiometer Regression Project

Author: Sheri Loftin
Date Created: July 29, 2020
Edited: Aug 12, 2020
Edited: April 7, 2021
"""
'''BLOCK 1'''
CUDA_VISIBLE_DEVICES = 0

#import tensorflow as tf
#from tensorflow.estimator import RunConfig
import keras
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.callbacks import Callback
from keras import Sequential 
from keras import layers
from keras.layers import LSTM, Dropout, Dense, BatchNormalization
from keras.layers import Bidirectional
from keras.preprocessing import sequence

# Helper libraries
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import math
import pandas as pd
import seaborn as sn
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.preprocessing import normalize
import random
import os

#print(device_lib.list_local_devices())
#print(tf.__version__)
#%%
"""
Read in data from Tim's radiometer model
"""
DirWins = 'D:/Radiometer/RNN_Study_9-3/Windows/'
# change to RadiometerData_7_4 for data including the sinusoid
# change to RadiometerData_no_sine for data without the sinusoid
 
Vhot_un = np.zeros((135000,33)) # Changed from 135000,33
Vcold_un = np.zeros((135000,33))
Vant_un = np.zeros((135000,33))
Thot_un = np.zeros((135000,33))
Tcold_un = np.zeros((135000,33))
Tant_un = np.zeros((135000,33))
train_wins = 100 #Changed from 100

idx = 0
# Increments for Cold, Ta Training
TTRefCold = [150., 170., 190., 210., 230., 250., 270., 290., 310., 330.]      
TTa = [180., 200., 220., 240., 260., 280., 300., 320., 340., 360., 380., 400., 420., 440., 460.]
TRefHot = 370.

# INcrements for b1, AR process
ARCoeff = [0.85, 0.9, 0.95]

# Increments for nVar
Nvarvals = [0.00001, 0.000005, 0.0000005]

# Testing gain of 1.0
gvl = 1.0                                               # Antennae Temp

# w1 = np.zeros([10,15,3,3])
# w3 = np.zeros([10,15,3,3])
# w5 = np.zeros([10,15,3,3])
# w11 = np.zeros([10,15,3,3])
# w33 = np.zeros([10,15,3,3])

# Loop through test 4 sets if test conditions
for trc in range(0,10):
    TRefCold = TTRefCold[trc]          # Vary the Cold Ref Temp 9 settings
    for tat in range(0,15):             # Vary the Ta temp 15 settings
        Ta = TTa[tat]                  
        for ac in range(0,3):          # Vary the AR coeff, b1 3 settings
            b1 = ARCoeff[ac]
            gvl = 1.0
            for nv in range(0,3):      # Vary the nVar
                nvar = Nvarvals[nv]

                nme = str(Ta)+"_"+str(TRefCold)+"_"+str(TRefHot)+"_"+str(b1)+"_"+str(nvar)+".txt"
    
                Vhot = np.loadtxt(DirWins + "ModelWin_VH_V3_"+nme, dtype=float)
                Vcold = np.loadtxt(DirWins + "ModelWin_VC_V3_"+nme, dtype=float)
                Vant = np.loadtxt(DirWins + "ModelWin_Va_V3_"+nme, dtype=float)
                Thot = np.loadtxt(DirWins + "ModelWin_TH_V3_"+nme, dtype=float)
                Tcold = np.loadtxt(DirWins + "ModelWin_TC_V3_"+nme, dtype=float)
                Tant = np.loadtxt(DirWins + "ModelWin_TrueTa_V3_"+nme, dtype=float)
                 
                Vhot_un[idx:idx+train_wins,:] = Vhot[0:train_wins,:]
                Vcold_un[idx:idx+train_wins,:] = Vcold[0:train_wins,:]
                Vant_un[idx:idx+train_wins,:] = Vant[0:train_wins,:]
                Thot_un[idx:idx+train_wins,:] = Thot[0:train_wins,:]
                Tcold_un[idx:idx+train_wins,:] = Tcold[0:train_wins,:]
                Tant_un[idx:idx+train_wins,:] = Tant[0:train_wins,:]
                idx = idx + train_wins
                #print ("Ingest Index: =", idx)

#%%
indices = np.arange(135000)
random.shuffle(indices)

Vhot_un = Vhot_un[indices]
Vcold_un = Vcold_un[indices]
Vant_un = Vant_un[indices]

Thot_un = Thot_un[indices]
Tcold_un = Tcold_un[indices]
Tant_un = Tant_un[indices]
#%%
plt.figure()
plt.style.use('seaborn-whitegrid')
plt.title('Hot, Cold, Antenna Voltage')
plt.plot(Vhot_un[1,:], color='red', label='Hot');
plt.plot(Vcold_un[1,:], color='blue', label='Cold');
plt.plot(Vant_un[1,:], color='green', label='Antenna');
plt.xlabel('Measurements')
plt.ylabel('voltage')
plt.legend()
plt.show()

#%%
''' Normalization style one '''

#print(Tant_un[1,:])
min_Tant = np.min(Tant_un)
max_Tant = np.max(Tant_un)
norm_Tant = (Tant_un - min_Tant)/(max_Tant - min_Tant)
#print(norm_Tant[1,:])

#print(Thot_un[1,:])
norm_Thot = (Thot_un - min_Tant)/(max_Tant - min_Tant)
#print(norm_Thot[1,:])

#print(Tcold_un[1,:])
norm_Tcold = (Tcold_un - min_Tant)/(max_Tant - min_Tant)
#print(norm_Tcold[1,:])

#print(Vant_un[1,:])
norm_Vant = (Vant_un - min_Tant)/(max_Tant - min_Tant)
#print(norm_Vant[1,:])

#print(Vhot_un[1,:])
norm_Vhot = (Vhot_un - min_Tant)/(max_Tant - min_Tant)
#print(norm_Vhot[1,:])

#print(Vcold_un[1,:])
norm_Vcold = (Vcold_un - min_Tant)/(max_Tant - min_Tant)
#print(norm_Vcold[1,:])

denorm_Tant = norm_Tant * (max_Tant - min_Tant) + min_Tant
#print(denorm_Tant[1,:])

denorm_Thot = norm_Thot * (max_Tant - min_Tant) + min_Tant
#print(denorm_Thot[1,:])

denorm_Tcold = norm_Tcold * (max_Tant - min_Tant) + min_Tant
#print(denorm_Tcold[1,:])

denorm_Vant = norm_Vant * (max_Tant - min_Tant) + min_Tant
#print(denorm_Vant[1,:])

denorm_Vhot = norm_Vhot * (max_Tant - min_Tant) + min_Tant
#print(denorm_Vhot[1,:])

denorm_Vcold = norm_Vcold * (max_Tant - min_Tant) + min_Tant
#print(denorm_Vcold[1,:])
#%%
''' Normalization style two '''

max_Thot = np.max(Thot_un)
norm_Tant = Tant_un/max_Thot
#print(norm_Tant[1,:])

#print(Thot_un[1,:])
norm_Thot = Thot_un/max_Thot
#print(norm_Thot[1,:])

#print(Tcold_un[1,:])
norm_Tcold = Tcold_un/max_Thot
#print(norm_Tcold[1,:])

#print(Vant_un[1,:])
norm_Vant = Vant_un/max_Thot
#print(norm_Vant[1,:])

print(Vhot_un[1,:])
norm_Vhot = Vhot_un/max_Thot
print(norm_Vhot[1,:])

#print(Vcold_un[1,:])
norm_Vcold = Vcold_un/max_Thot
#%%
'''
Test/Train split after normalization (for comparison)
Setup ModelWin data for the RNN code below
Run this and then skip to the RNN CODE BLOCK to run the RNN code
'''
X = np.array([norm_Vhot, norm_Vcold, norm_Vant, norm_Thot, norm_Tcold])
X = X.transpose(1, 2, 0)
     
y = norm_Tant[:,16]
     
print(y.shape, X.shape)
     
X_train = X[0:99999, :]
X_test = X[100000:135000, :]
     
y_train = y[0:99999]
y_test = y[100000:135000]

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


#%%
'''
Setting up the RNN using LSTM
'''
epoch = 1000 #previously 2500
#log_dir='D:/Radiometer/RNN_Study_9-3/tensorboard/'

print('Build model...')
regressor = keras.Sequential()
regressor.add(LSTM((1), batch_input_shape=(None,33,5), return_sequences=True))
regressor.add(BatchNormalization())
regressor.add(Dropout(0.25))
regressor.add(LSTM((1), return_sequences=False))

opt = keras.optimizers.Adam(lr=0.001, decay=1e-6)
regressor.compile(optimizer = opt, loss = 'mse', metrics =['accuracy', 'mae'])

#regressor.summary()

#filepath = "RNN-{epoch:02d}-{accuracy:.3f}"
#filepath = "RNN-{epoch:02d}"
# checkpointer = ModelCheckpoint("D:/Radiometer/RNN_Study_9-3/models/{}.model".format(filepath), 
#                                 monitor='accuracy',
#                                 verbose=0, 
#                                 save_best_only=True)
# tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=100000)

# history = regressor.fit(X_train, y_train, epochs = epoch, 
#                           batch_size = 33, verbose= True, 
#                           validation_data=(X_test,y_test),
#                           callbacks=[checkpointer])
history = regressor.fit(X_train, y_train, epochs = epoch, 
                          batch_size = 33, verbose= True, 
                          validation_data=(X_test,y_test))
regressor.summary()

#%%
regressor.save("D:/Radiometer/RNN_Study_9-3/modelRNN_RMSE_0_.h5")
print("Saved model to disk")
 
#%%
'''
RMSE and MAE calculation
'''
ypred = regressor.predict(X_test)
rmse = np.sqrt(metrics.mean_squared_error(y_test, ypred))
print("RMSE= ", rmse)

mae = metrics.mean_absolute_error(y_test, ypred)
print("MAE: ", mae)
# plt.figure()
# plt.scatter(range(35000),ypred,c='r')
# plt.scatter(range(35000),y_test,c='g')
# plt.show()

#%%
'''
Reversing the normalization equation
denorm_Tant = norm_Tant * (max_Tant - min_Tant) + min_Tant
#print(denorm_Tant[1,:])
'''

#denorm_ypred = ypred * (max_Tant - min_Tant) + min_Tant
#print(denorm_ypred.shape)

#denorm_ytest = y_test * (max_Tant - min_Tant) + min_Tant
#print(denorm_ytest.shape)

denorm_ypred = ypred * max_Thot
denorm_ytest = y_test * max_Thot

check = np.sqrt(metrics.mean_squared_error(denorm_ytest, denorm_ypred))
print(check)
check2 = metrics.mean_absolute_error(denorm_ytest, denorm_ypred)
print("MAE: ", check2)

plt.figure()
plt.style.use('seaborn-whitegrid')
plt.title('LSTM Regression with 33 Measurements/Window and RMSE error bars')
plt.errorbar(denorm_ytest, denorm_ypred, yerr=check, fmt='o', color='black',
             ecolor='gray', elinewidth=3, capsize=0);
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()



