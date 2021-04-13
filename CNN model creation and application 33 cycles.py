# -*- coding: utf-8 -*-
"""
Created on Thu May 28 08:57:50 2020

@author: saloftin

Cleaned up version of the CNN Radiometer project focusing on 33 cycles/window
and the expanded range of values for the antenna temp.

Latest update: July 4, 2020
"""


# Helper libraries
import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt
import random

# Dataset prep and initial regression libraries
from sklearn import metrics

# Neural network libraries
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Activation
from keras.layers import LSTM, BatchNormalization
from sklearn.metrics import mean_squared_error
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

#%%
"""
Read in data from Tim's radiometer model
Each file contains 10,000 sets of 51 measurements each
There are 6 files per dataset and 4 versions (A,B,C,D) of each set
"""

Vhot_un = np.zeros((48000,51))
Vcold_un = np.zeros((48000,51))
Vant_un = np.zeros((48000,51))
Thot_un = np.zeros((48000,51))
Tcold_un = np.zeros((48000,51))
Tant_un = np.zeros((48000,51))
train_wins = 1000

idx = 0
TRefHot = 370.                                        # Hot Ref Temp
TTRefCold = [170, 200, 230, 260, 290, 320]      
TTa = [180, 220, 260, 300, 340, 380, 420, 460]
GScl = [0.9, 1.0, 1.10] 

w1 = np.zeros([6,8])
w3 = np.zeros([6,8])
w5 = np.zeros([6,8])
w11 = np.zeros([6,8])
w33 = np.zeros([6,8])

for trc in range(0,6):
    TRefCold = TTRefCold[trc]                                 # Cold Ref Temp
    for tat in range(0,8):
        Ta = TTa[tat]
        for gval in range(1,2):
            gvl = GScl[gval]

            nme = str(gvl)+"_"+str(Ta)+"_"+str(TRefCold)+"_"+str(TRefHot)+".txt"

            Vhot = np.loadtxt("new_data_6_5/RadiometerData/ModelWin_VH_V2_"+nme, dtype=float)
            Vcold = np.loadtxt("new_data_6_5/RadiometerData/ModelWin_VC_V2_"+nme, dtype=float)
            Vant = np.loadtxt("new_data_6_5/RadiometerData/ModelWin_Va_V2_"+nme, dtype=float)
            Thot = np.loadtxt("new_data_6_5/RadiometerData/ModelWin_TH_V2_"+nme, dtype=float)
            Tcold = np.loadtxt("new_data_6_5/RadiometerData/ModelWin_TC_V2_"+nme, dtype=float)
            Tant = np.loadtxt("new_data_6_5/RadiometerData/ModelWin_Ta_V2_"+nme, dtype=float)
             
            Vhot_un[idx:idx+train_wins,:] = Vhot[0:train_wins,:]
            Vcold_un[idx:idx+train_wins,:] = Vcold[0:train_wins,:]
            Vant_un[idx:idx+train_wins,:] = Vant[0:train_wins,:]
            Thot_un[idx:idx+train_wins,:] = Thot[0:train_wins,:]
            Tcold_un[idx:idx+train_wins,:] = Tcold[0:train_wins,:]
            Tant_un[idx:idx+train_wins,:] = Tant[0:train_wins,:]
            idx = idx + train_wins
            #print ("Ingest Index: =", idx)


indices = np.arange(48000)
random.shuffle(indices)

Vhot_un = Vhot_un[indices]
Vcold_un = Vcold_un[indices]
Vant_un = Vant_un[indices]

Thot_un = Thot_un[indices]
Tcold_un = Tcold_un[indices]
Tant_un = Tant_un[indices]


print("Vhot: ", Vhot_un.shape)
print("Vcold: ", Vcold_un.shape)
print("Vant: ", Vant_un.shape)
print("Thot: ", Thot_un.shape)
print("Tcold: ", Tcold_un.shape)
print("Tant: ", Tant_un.shape)

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

"""
Create data matrix from Tim's model data since identification of the 
individual parameters Vhot, Vcold, etc... are not necessary to our purposes.

The datamatrix contains all the parameters except Tant.
data_Tant is the separate array we wish to predict using only the middle value
of each 33 value window.

Vhot: 0-33
Vcold: 33-66
Vant: 66-99
Thot: 99-132
Tcold: 132-165
"""    

data_matrix = np.concatenate((Vhot_un[:,0:33], Vcold_un[:,0:33], Vant_un[:,0:33], 
                              Thot_un[:,0:33], Tcold_un[:,0:33]), axis=1) 

data_Tant = Tant_un[:,17]

print(data_matrix.shape, data_Tant.shape)

#%%  
""" 
Plot values
"""
plt.figure()
plt.scatter(data_matrix[:,83], data_Tant[:])
plt.xlabel('Temp Antenna')
plt.ylabel('Voltage Antenna')
plt.show()

plt.figure()
plt.scatter(data_matrix[:,36], data_matrix[:,135])
plt.xlabel('Temp Cold')
plt.ylabel('Voltage Cold')
plt.show()

plt.figure()
plt.scatter(data_matrix[:,5], data_matrix[:,104])
plt.xlabel('Temp Hot')
plt.ylabel('Voltage Hot')
plt.show()

#%%
"""
Normalization using tanh code
Source: https://stackoverflow.com/questions/43061120/tanh-estimator-normalization-in-python
Note this code implements a modified tanh-estimator proposed in Efficient 
approach to Normalization of Multimodal Biometric Scores, 2011
"""
# #Vhot
# VAll = [Vhot_un, Vcold_un, Vant_un] 
# mV = np.mean(VAll, axis=0) 
# stdV = np.std(VAll, axis=0) 
# Vhot = 0.5 * (np.tanh(0.01 * ((Vhot_un - mV) / stdV)) + 1) 

# #Vcold 
# Vcold = 0.5 * (np.tanh(0.01 * ((Vcold_un - mV) / stdV)) + 1) 
# #Vant 
# Vant = 0.5 * (np.tanh(0.01 * ((Vant_un - mV) / stdV)) + 1) 

# #Thot 
# TAll = [Thot_un, Tcold_un, Tant_un] 
# mT = np.mean(TAll, axis=0) 
# stdT = np.std(TAll, axis=0) 
# Thot = 0.5 * (np.tanh(0.01 * ((Thot_un - mT) / stdT)) + 1) 

# #Tcold 
# Tcold = 0.5 * (np.tanh(0.01 * ((Tcold_un - mT) / stdT)) + 1) 

# #Tant 
# Tant = 0.5 * (np.tanh(0.01 * ((Tant_un - mT) / stdT)) + 1) 

#%%
'''
CNN REGRESSION
'''
#%%
''' 33 MEASUREMENTS/WINDOW
Implementing a CNN on the Radiometer data
Code source: https://www.datatechnotes.com/2019/12/how-to-fit-regression-data-with-cnn.html

Reminder of our data structure
33 measurements/window predicting the center value of Ant Temp

y = data_Tant # Antenna Temp array with shape(60000, 1)
X = data_matrix #Hot, Cold, Ant Voltages and Hot, Cold Temp matrix with 
shape (60000, 5, 33)
'''

X = np.array([Vhot_un[:,0:33], Vcold_un[:,0:33], Vant_un[:,0:33], 
                              Thot_un[:,0:33], Tcold_un[:,0:33]])
X = X.transpose(1, 0, 2)

y = Tant_un[:,17]

print(y.shape, X.shape)

X_train = X[0:99999, :]
X_test = X[100000:135000, :]
     
y_train = y[0:99999]
y_test = y[100000:135000]

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

#%%
rmse = 2
while rmse > 0.4:
    model = Sequential()
    model.add(Conv1D(32, 5, strides=5, activation="relu", input_shape=(5, 33)))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='relu'))
    model.compile(loss="mse", optimizer="adam", metrics=["acc", "mse"])
    
    checkpointer = ModelCheckpoint(filepath="best_weights.hdf5", 
                                   monitor = 'acc',
                                   verbose=1, 
                                   save_best_only=True)
    
    model.fit(X_train, y_train, batch_size=12,epochs=100, verbose=0, 
              validation_data=(X_test, y_test))

    ypred = model.predict(X_test)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, ypred))
    print("RMSE = ", rmse)

print(model.evaluate(X_train, y_train))
#model_weights = model.save_weights()

print("RMSE= ", rmse)

mae = metrics.mean_absolute_error(y_test, ypred)
print("MAE: ", mae)

#%%
'''
Serialize model to use in future
Code source: https://machinelearningmastery.com/save-load-keras-deep-learning-models/
'''

# save model and architecture to single file
model.save("model33_simple_RMSE_0.3941.h5")
print("Saved model to disk")
#%%
'''Visualizing the results
Code source: https://towardsdatascience.com/visualizing-intermediate-activation-in-convolutional-neural-networks-with-keras-260b36d60d0
'''

print('Error assessments for CNN prediction method with 33 measurements per window: ')
print('MAE = ', metrics.mean_absolute_error(y_test, ypred))
print('MSE = ', metrics.mean_squared_error(y_test, ypred))
print('RMSE = ', np.sqrt(metrics.mean_squared_error(y_test, ypred)))

rmse = np.sqrt(metrics.mean_squared_error(y_test, ypred))

plt.style.use('seaborn-whitegrid')
plt.title('CNN Regression with 33 Measurements/Window and RMSE error bars')
plt.errorbar(y_test, ypred, yerr=rmse, fmt='o', color='black',
             ecolor='gray', elinewidth=3, capsize=0);
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()

hist_dict = model.history.history
hist_dict.keys()

acc = model.history.history['acc']
val_acc = model.history.history['val_acc']
loss = model.history.history['loss']
val_loss = model.history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training MSE')
plt.plot(epochs, val_acc, 'b', label='Validation acc', color='red')

plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss', color='red')
plt.title('Training and validation loss')
plt.ylim(0,50)

plt.legend()
plt.show()

#%%
'''
Load model from previous CNN runs
Code source: https://machinelearningmastery.com/save-load-keras-deep-learning-models/
32 measurements/window
'''
# load model
model33 = load_model('model33_Tim_RMSE_1.9589_7_6.h5')
# summarize model.
model33.summary()
#%%
"""
Read in data from Tim's radiometer model
Each file contains 999,999 measurements each
There are 6 files per dataset and 6 versions (A,B,E,F,G,H) of each set
"""

DirWins = 'C:/Users/saloftin/Radiometer/new_data_7_4/Cycles_7_4/'

Vhot_data = []
Vcold_data = []
Vant_data = []
Thot_data = []
Tcold_data = []
Tant_data = []
# #train_wins = 1000

idx = 0
TTRefCold = [150., 170., 190., 210., 230., 250., 270., 290., 310., 330.]      
TTa = [180., 200., 220., 240., 260., 280., 300., 320., 340., 360., 380., 400., 420., 440., 460.]
TRefHot = 370.
# INcrements for b1, AR process
ARCoeff = [0.85, 0.9, 0.95]
Nvarvals = [0.00001, 0.000005, 0.0000005]

w1 = np.zeros([6,8])
w3 = np.zeros([6,8])
w5 = np.zeros([6,8])
w11 = np.zeros([6,8])
w33 = np.zeros([6,8])

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

            Vhot = np.loadtxt(DirWins+"/ModelCycles_VHot_V3_"+nme, dtype=float)
            Vcold = np.loadtxt(DirWins+"/ModelCycles_VCold_V3_"+nme, dtype=float)
            Vant = np.loadtxt(DirWins+"/ModelCycles_Va_V3_"+nme, dtype=float)
            Thot = np.loadtxt(DirWins+"/ModelCycles_THot_V3_"+nme, dtype=float)
            Tcold = np.loadtxt(DirWins+"/ModelCycles_TCold_V3_"+nme, dtype=float)
            Tant = np.loadtxt(DirWins+"/ModelCycles_TrueTa_V3_"+nme, dtype=float)
             
            Vhot_data = np.append(Vhot_data, Vhot[0:20000,], axis=0)
            Vcold_data = np.append(Vcold_data, Vcold[0:20000,], axis=0)
            Vant_data = np.append(Vant_data, Vant[0:20000,], axis=0)
            Thot_data = np.append(Thot_data, Thot[0:20000,], axis=0)
            Tcold_data = np.append(Tcold_data, Tcold[0:20000,], axis=0)
            Tant_data = np.append(Tant_data, Tant[0:20000,], axis=0)
            #print ("Ingest Index: =", idx)


print("Vhot: ", Vhot_data.shape)
print("Vcold: ", Vcold_data.shape)
print("Vant: ", Vant_data.shape)
print("Thot: ", Thot_data.shape)
print("Tcold: ", Tcold_data.shape)
print("Tant: ", Tant_data.shape)

data = np.asmatrix((Vhot_data, Vcold_data, Vant_data, Thot_data, Tcold_data), dtype = float)
print(data.shape)

#%%
'''
Loop through the data to apply the model to consecutive windows
'''
size = 33
meas = data.shape[1]
limit = meas - size
Y = np.zeros(limit)
X = np.zeros((limit, 5, size))

for i in range(0, limit):
    j = i + size
    k = i + 17
    X33 = data[:, i:j]
    Y[i] = Tant_data[k]
    X[i,:,:] = X33

print(X.shape)

preds = model33.predict(X)


#%%
'''Visualization of the results
'''
print('Error assessments for CNN prediction method with 33 measurements per window: ')
print('MAE = ', metrics.mean_absolute_error(Y, preds))
print('MSE = ', metrics.mean_squared_error(Y, preds))
print('RMSE = ', np.sqrt(metrics.mean_squared_error(Y, preds)))

rmse = np.sqrt(metrics.mean_squared_error(Y, preds))

plt.style.use('seaborn-whitegrid')
plt.title('CNN Regression with 33 Measurements/Window and RMSE error bars')
plt.errorbar(Y, preds, yerr=rmse, fmt='o', color='black',
             ecolor='cyan', elinewidth=3, capsize=0);
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()


