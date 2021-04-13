# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 12:12:57 2020

@author: saloftin
"""

import csv
import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Dataset prep and initial regression libraries
from sklearn import metrics
from sklearn import linear_model
import scipy
from scipy import stats

# Neural network libraries
from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, Flatten, Activation
from keras.layers import LSTM, BatchNormalization
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import PolynomialFeatures
from tensorflow.keras.optimizers import Adam, Nadam, Adamax, Adagrad, Adadelta, SGD
import astropy
from astropy.modeling.models import Lorentz1D
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel

#setup location and arrays for data
DataDir = 'D:/Radiometer/Gordons_project/Vary_Sigma_only/Less_noise/Zero/Data/'   # Data file storage location

#%%
'''
Create data for model using Gordon's process and requirements
Removed for example sharing purposes as this isn't my code
'''

#%%

'''
load in data into a matrix of shape (100,2) with each entry containing an array of 41 values
'''
num=50000    # number of samples to read in
Truth = np.zeros((num,61))
Log = np.loadtxt(DataDir+"Vary_Signal_Peak")
Width = np.loadtxt(DataDir+"Vary_Sigma")
Signal = np.zeros((num,61))

for i in range(num):
    inc = str(i)
    T = np.loadtxt(DataDir+"Vary_Sigma_Truth"+inc)
    Truth[i] = T
    S = np.loadtxt(DataDir+"Vary_Sigma_Signal"+inc)
    Signal[i,:] = S
    #print(inc)
    
print('done')


Truth_new = Truth.reshape(num, 1, 61)
Signal_new = Signal.reshape(num, 1, 61)
print(Log.shape, Width.shape, Signal_new.shape, Truth_new.shape)
print(Width.shape, Signal_new.shape, Truth_new.shape)

#%%
''' Plotting Data '''
time = [x for x in range(1000)]

plt.figure()
plt.style.use('seaborn-whitegrid')
#plt.ylim([-2,12])
plt.title('Plot of Sigma: first 1000 values')
plt.scatter(time, Width[0:1000], color='red', label='Width')            
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend(loc='lower right')
plt.show()

plt.figure()
plt.ylim([0,1050])
plt.title('Plot of Signal Peak: first 1000 values')        
plt.scatter(time, Log[0:1000], color='black', label='Log Opacity')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend(loc='lower right')
plt.show()
#%%
''' Plotting the noisy signal vs the true signal '''

s = Signal_new[10,:]
s0 = s.reshape(61)
t = Truth_new[10,:]
t0 = t.reshape(61)

fig, ax = plt.subplots()
ax.grid(True)
ax.set_ylim(970,1010)
ax.set_xlim(-10,10)
plt.subplots_adjust(left= .15, bottom=0.30)

freq = np.arange(-15.0, 15.5, 0.5) # Frequency range -10 to 10 MHz 
l, = plt.step( freq, t0, lw = 2, where ='mid', label='Truth')
m, = plt.plot( freq, s0, linestyle = ':', lw=3, label='Signal')

plt.title('SSOLVE Observation', fontsize = 18)
plt.ylabel('T (K)', fontsize = 12)
plt.xlabel('Offset Freq (MHz)', fontsize = 10)
plt.legend(loc='lower right')
ax.margins(x=1)

plt.show()

#%%
'''Test-Train Split'''
num=50000
train = num*.75
train = int(train)
test = num-train

#Truth3 = Truth.reshape(num,2,1)

X_train = Signal_new[0:train, :, :]
X_test = Signal_new[train:num, :, :]

y_train = Truth_new[0:train, :, :]
y_test = Truth_new[train:num, :, :]

yL_train = Log[0:train]
yL_test = Log[train:num]
yW_train = Width[0:train]
yW_test = Width[train:num]

print(yL_train.shape, yW_train.shape, X_train.shape, y_train.shape)
print(yL_test.shape, yW_test.shape, X_test.shape, y_test.shape)
print(yW_train.shape, X_train.shape, y_train.shape)
print( yW_test.shape, X_test.shape, y_test.shape)

#%%
'''CNN predicting Log Opacity'''
rmseL = 3
while (rmseL > 2):
    Lmodel = Sequential()
    Lmodel.add(Conv1D(64, 1, activation="relu", input_shape=(1,61)))
    Lmodel.add(Flatten())
    Lmodel.add(Dense(100, activation="relu"))
    Lmodel.add(Dense(50, activation="relu"))
    Lmodel.add(Dense(25, activation="relu"))
    Lmodel.add(Dense(1))
    Lmodel.compile(loss="mse", optimizer="adagrad")
    Lmodel.fit(X_train, yL_train, batch_size=12,epochs=2000, verbose=0)
     
    yLpred = Lmodel.predict(X_test)
    print(Lmodel.evaluate(X_train, yL_train))
    #Lmodel_weights = Lmodel.save_weights()
    rmseL = np.sqrt(metrics.mean_squared_error(yL_test, yLpred))
    print("Log Opacity MSE: %.4f" % mean_squared_error(yL_test, yLpred))
    print("Log Opacity RMSE: %.4f" % rmseL) 


# save model and architecture to single file
Lmodel.save("D:/Radiometer/Gordons_project/Vary_Peak_Sigma/Full_Noise/Figures/CNN_model_Feb12_Signal_Peak_0.h5")
print("Saved model to disk")

yLtest = yL_test.reshape(12500, 1)

# scipy.stats.chisquare(yLtest,yLpred,ddof=2,axis=0)
#%%
'''CNN predicting Width'''
rmseW = 3
while (rmseW > 0.1):
    Wmodel = Sequential()
    Wmodel.add(Conv1D(64, 1, activation="relu", input_shape=(1,61)))
    Wmodel.add(Conv1D(64, 1, activation="relu", input_shape=(1,61)))
    Wmodel.add(Conv1D(64, 1, activation="relu", input_shape=(1,61)))
    Wmodel.add(Dense(100, activation="relu"))
    Wmodel.add(Dense(50, activation="relu"))
    Wmodel.add(Dense(25, activation="relu"))
    Wmodel.add(Flatten())
    Wmodel.add(Dense(1))
    Wmodel.compile(loss="mse", optimizer="adagrad")
    Wmodel.fit(X_train, yW_train, batch_size=61,epochs=2000, verbose=0) 
    yWpred = Wmodel.predict(X_test)
     
    print(Wmodel.evaluate(X_train, yW_train))
    #Lmodel_weights = Lmodel.save_weights()
    rmseW = np.sqrt(metrics.mean_squared_error(yW_test, yWpred))
    print("Width MSE: %.4f" % mean_squared_error(yW_test, yWpred))
    print("Width RMSE: %.4f" % rmseW)

# save model and architecture to single file
Wmodel.save("D:/Radiometer/Gordons_project/Vary_Sigma_only/Less_noise/Zero/Figures/model_Mar23_Sigma_0.h5")
print("Saved model to disk")
#%%
yWtest = yW_test.reshape(12500, 1)
scipy.stats.chisquare(yWtest,yWpred,ddof=2,axis=0)

#%%
''' Log Opacity Plot '''

plt.style.use('seaborn-whitegrid')
plt.title('CNN Regression of Signal Peak with 61 Measurements and RMSE error bars')
plt.errorbar(yLpred, yL_test, yerr=rmseL, fmt='o', color='red',
              ecolor='cyan', elinewidth=3, capsize=0, label='Predicted Values with RMSE error');
#plt.plot(test_lo, test_w, 's', color='black', label='True Values')
plt.xlabel('Predicted Signal Peak')
plt.ylabel('True Signal Peak')
plt.legend(loc='lower right')
plt.show()

#%%
''' Log Opacity - prediction vs truth '''
time = [x for x in range(1000)]

plt.style.use('seaborn-whitegrid')
plt.title('Signal Peak: Predicted vs True Values')
plt.scatter(time, yLpred[0:1000], color='red', label='Predicted')            
plt.scatter(time, yL_test[0:1000], color='black', label='True Value')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend(loc='lower right')
plt.show()
#%%
''' Width Plot '''

plt.figure()
plt.style.use('seaborn-whitegrid')
plt.ylim([-2.5,6])
plt.title('CNN Regression of Sigma with 61 Measurements and RMSE error bars')
plt.errorbar(yWpred, yW_test, yerr=rmseW, fmt='o', color='red',
             ecolor='cyan', elinewidth=3, capsize=0, label='Predicted Values with RMSE error');
#plt.plot(test_lo, test_w, 's', color='black', label='True Values')
plt.xlabel('Predicted Sigma')
plt.ylabel('True Sigma')
plt.legend(loc='lower right')
plt.show()

#%%
''' Width - prediction vs truth '''
time = [x for x in range(1000)]
plt.figure()
plt.style.use('seaborn-whitegrid')
plt.ylim([-1,5])
plt.title('Sigma: Predicted vs True Values')
plt.scatter(time, yWpred[0:1000], color='red', label='Predicted')            
plt.scatter(time, yW_test[0:1000], color='black', label='True Value')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend(loc='lower right')
plt.show()

#%%
''' Blending the Log Opacity into the signal array '''
     
X = np.concatenate((Signal, yWpred), axis=1)
print(X.shape)
XFull = X.reshape(num, 1, 62)


XFull_train = XFull[0:train]
XFull_test = XFull[train:num]

yW_train = Width[0:train]
yW_test = Width[train:num]

print(XFull_train.shape, yW_train.shape)
print(XFull_test.shape, yW_test.shape)
#%%
''' Regressing sigma while including the peak signal '''
mseS = 4
while mseS > .9:
    Smodel = Sequential()
    Smodel.add(Conv1D(128, 1, activation="relu", input_shape=(1,62)))
    Smodel.add(Conv1D(64, 1, activation="relu", input_shape=(1,62)))
    Smodel.add(Conv1D(64, 1, activation="relu", input_shape=(1,62)))
    Smodel.add(Dense(150, activation="relu"))
    Smodel.add(Dense(50, activation="relu"))
    Smodel.add(Dense(25, activation="relu"))
    Smodel.add(Flatten())
    Smodel.add(Dense(1))
    Smodel.compile(loss="mse", optimizer="adagrad")
    Smodel.fit(XFull_train, yW_train, batch_size=12,epochs=2000, verbose=0) 
    ySpred = Smodel.predict(XFull_test)

    mse =  Smodel.evaluate(XFull_train, yW_train)
    print('MSE: %.4f' % mse)
    rmseS = np.sqrt(metrics.mean_squared_error(yW_test, ySpred))
    print("Width MSE: %.4f" % mean_squared_error(yW_test, ySpred))
    print("Width RMSE: %.4f" % rmseS)

Smodel.save("D:/Radiometer/Gordons_project/Vary_Peak_Sigma/Full_Noise/Figures/CNN_model_Feb11_Full_Signal_1.h5")
print("Saved model to disk")
#%%
s = ySpred[0,:]
s0 = s.reshape(41)
t = Truth_new[0,:]
t0 = t.reshape(41)

fig, ax = plt.subplots()
ax.grid(True)
ax.set_ylim(940,1020)
ax.set_xlim(-10,10)
plt.subplots_adjust(left= .15, bottom=0.30)

freq = np.arange(-10.0, 10.5, 0.5) # Frequency range -10 to 10 MHz 
l, = plt.plot( freq, t0, lw = 2, label='Truth')
m, = plt.plot( freq, s0, lw=3, label='Prediction')

plt.title('SSOLVE Observation vs CNN Prediction', fontsize = 18)
plt.ylabel('T (K)', fontsize = 12)
plt.xlabel('Offset Freq (MHz)', fontsize = 10)
plt.legend(loc='lower right')
ax.margins(x=1)

plt.show()
#%%
''' Linear - Nonlinear - MLP regression attempts  '''

#combining Sigma and Peak signam into one array
combined = np.vstack((Log, Width)).T
print(combined.shape)

#%%
'''Linear/Logistic Regression Test-Train Split'''
train = num*.75
train = int(train)
test = num-train

X_train = Signal_new[0:train, :, :]
X_test = Signal_new[train:num, :, :]

y_train = combined[0:train,:]
y_test = combined[train:num,:]

print(y_train.shape, X_train.shape)
print(y_test.shape, X_test.shape)

#%%
""" Linear Regression
Regression predictions of both Log Opacity and Width: RMSE of 1.8515
Regression predictions of just Width: RMSE of 2.5219
Regression predictions of just Log Opacity: RMSE of 0.7923
"""
'''Test-Train Split'''
train = num*.75
train = int(train)
test = num-train

# Linear Regression
X_train = Signal[0:train, :]
X_test = Signal[train:num, :]

# y_trainL = Truth[0:train, :]
# y_testL = Truth[train:num, :]

# print(y_trainL.shape, X_trainL.shape)
# print(y_testL.shape, X_testL.shape)
#Linear regression attempt
# fit a model
lm = linear_model.LinearRegression()
lm_model = lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)

rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print(rmse)

#%%
n = 0 # N. . .
pred_lo= [x[n] for x in y_pred]
test_lo = [x[n] for x in y_test]
i=1
pred_w= [x[i] for x in y_pred]
test_w = [x[i] for x in y_test]

rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

plt.figure()
plt.style.use('seaborn-whitegrid')
plt.title('Linear Regression of Signal Peak with 61 Measurements and RMSE error bars')
plt.errorbar(pred_lo, test_lo, fmt='o', color='red', capsize=0);
#plt.plot(test_lo, test_w, 's', color='black', label='True Values')
plt.xlabel('Signal Peak prediction')
plt.ylabel('Signal Peak Truth')
#plt.legend(loc='lower right')
plt.show()

plt.figure()
plt.style.use('seaborn-whitegrid')
plt.title('Linear Regression of Sigma with 61 Measurements and RMSE error bars')
plt.errorbar(pred_w, test_w, fmt='o', color='red', capsize=0);
#plt.plot(test_lo, test_w, 's', color='black', label='True Values')
plt.xlabel('Signal Peak prediction')
plt.ylabel('Signal Peak Truth')
#plt.legend(loc='lower right')
plt.show()


#%%
'''MLP regressing to the tuple (Peak Signal, Width) '''
 
mlp = Sequential()
mlp.add(Dense(350, input_shape=(1,61), activation='relu'))
mlp.add(Dense(50, activation='relu'))
mlp.add(Flatten())
mlp.add(Dense(2, activation='relu'))

mlp.compile(loss='mse', optimizer='adagrad', metrics=['mse'])
mlp.fit(X_train, y_train, epochs=1000, batch_size=30, verbose=0, validation_split=0.2)

mlp_results = mlp.evaluate(X_test, y_test, verbose=1)
print(f'Log Opacity Test results - Loss: {mlp_results[0]} - MSE: {mlp_results[1]}')
# save model and architecture to single file
mlp.save("D:/Radiometer/Gordons_project/Vary_Peak_Sigma/Full_noise/Figures/mlp_model_Feb16_tuple.h5")
print("Saved model to disk")

#%%

n = 0 # N. . .
pred_peak = [x[n] for x in y_pred]
test_peak = [x[n] for x in y_test]
i=1
pred_sigma = [x[i] for x in y_pred]
test_sigma = [x[i] for x in y_test]

#%%
''' Peak signal Plot '''

plt.figure()
plt.style.use('seaborn-whitegrid')
plt.title('MLP Regression of Signal Peak with 61 Measurements')
plt.errorbar(pred_peak, test_peak, yerr=rmseL, fmt='o', color='red',
             ecolor='cyan', elinewidth=3, capsize=0, label='Predicted Values with RMSE error');
#plt.plot(test_lo, test_w, 's', color='black', label='True Values')
plt.xlabel('Predicted Peak Signal')
plt.ylabel('True Peak Signal')
plt.legend(loc='lower right')
plt.show()

#%%
''' Peak Signal - prediction vs truth '''
time = [x for x in range(1000)]

plt.figure()
plt.style.use('seaborn-whitegrid')
#plt.ylim([-7,2])
plt.title('Peak Signal: Predicted vs True Values')
plt.scatter(time, pred_peak[0:1000], color='red', label='Predicted')            
plt.scatter(time, test_peak[0:1000], color='black', label='True Value')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend(loc='lower right')
plt.show()
#%%
''' Width Plot '''

plt.figure()
plt.style.use('seaborn-whitegrid')
plt.title('MLP Regression of Sigma with 61 Measurements')
plt.errorbar(pred_sigma, test_sigma, yerr=rmseW, fmt='o', color='red',
             ecolor='cyan', elinewidth=3, capsize=0, label='Predicted Values with RMSE error');
#plt.plot(test_lo, test_w, 's', color='black', label='True Values')
plt.xlabel('Predicted Sigma')
plt.ylabel('True Sigma')
plt.legend(loc='lower right')
plt.show()

#%%
''' Sigma - prediction vs truth '''
time = [x for x in range(1000)]
plt.figure()
plt.style.use('seaborn-whitegrid')
plt.title('Sigma: Predicted vs True Values')
plt.scatter(time, pred_sigma[0:1000], color='red', label='Predicted')            
plt.scatter(time, test_sigma[0:1000], color='black', label='True Value')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend(loc='lower right')
plt.show()
#%%
# Width
mlpW = Sequential()
mlpW.add(Dense(350, input_shape=(1,41), activation='relu'))
mlpW.add(Dense(50, activation='relu'))
mlpW.add(Flatten())
mlpW.add(Dense(1, activation='relu'))

mlpW.compile(loss='mse', optimizer='adam', metrics=['mse'])
mlpW.fit(X_train, yW_train, epochs=1000, batch_size=30, verbose=0, validation_split=0.2)

mlpW_results = mlpW.evaluate(X_test, yW_test, verbose=1)
print(f'Width Test results - Loss: {mlpW_results[0]} - MSE: {mlpW_results[1]}')
# save model and architecture to single file
mlpW.save("D:/Radiometer/Gordons_project/Vary_Log_Width/Less_noise/Figures/mlp_model_quarter_noise_JAN10_Width_2_.h5")
print("Saved model to disk")
#%%
''' Plotting lost figures
Allows loading of data and previous model to plot results
'''
DataDir = 'D:/Radiometer/Gordons_project/Vary_Log_Width/Limited_range_W_0-5_L_4-2/'   # Data file storage location

# Read in data desired to apply the model to
num=100000    # number of samples to read in
Truth = np.zeros((num,2))
Log = np.loadtxt(DataDir+"Data/Vary_Log_Opac")
Width = np.loadtxt(DataDir+"Data/Vary_Width")
Signal = np.zeros((num,41))

for i in range(num):
    inc = str(i)
    T = (Log[i],Width[i])
    Truth[i] = T
    S = np.loadtxt(DataDir+"Data/Vary_Log_Opac_Signal"+inc)
    Signal[i,:] = S
       
Signal1 = Signal.reshape(num, 41, 1)
Signal_new = Signal1.reshape(num, 1, 41)
print(Log.shape, Width.shape, Signal_new.shape)

'''Test-Train Split'''

train = num*.75
train = int(train)
test = num-train

#Truth3 = Truth.reshape(num,2,1)

X_train = Signal_new[0:train, :, :]
X_test = Signal_new[train:num, :, :]

yL_train = Log[0:train]
yL_test = Log[train:num]
yW_train = Width[0:train]
yW_test = Width[train:num]

print(yL_train.shape, yW_train.shape, X_train.shape)
print(yL_test.shape, yW_test.shape, X_test.shape)

# Load model and apply to data
mlL = load_model(DataDir+'/Figures/model_DEC28_LogOpac_0_5778.h5') # loading the Log Opacity model

Lpred = mlL.predict(X_test) # applying the model for Log Opacity to the dataset
     
print(mlL.evaluate(X_train, yL_train)) # finding the error rate of Log Opacity with the model application

rmseL = np.sqrt(metrics.mean_squared_error(yL_test, Lpred))
print("Log Opacity MSE: %.4f" % mean_squared_error(yL_test, Lpred))
print("Log Opacity RMSE: %.4f" % rmseL)

# Load model and apply to data
mlW = load_model(DataDir+'/Figures/model_DEC28_Width_1_4091.h5') # loading the Width model
Wpred = mlW.predict(X_test) # applying the model for Width to the dataset
     
print(mlW.evaluate(X_train, yW_train)) # finding the error rate of Width with the model application

rmseW = np.sqrt(metrics.mean_squared_error(yW_test, Wpred))
print("Width MSE: %.4f" % mean_squared_error(yW_test, Wpred))
print("Width RMSE: %.4f" % rmseW)
#%%
''' Log Opacity Plot '''

plt.style.use('seaborn-whitegrid')
plt.title('CNN Regression of Log Opacity (range -4 - -2) with 41 Measurements')
plt.errorbar(Lpred, yL_test, yerr=rmseL, fmt='o', color='red',
             ecolor='cyan', elinewidth=3, capsize=0, label='Predicted Values with RMSE error');
#plt.plot(test_lo, test_w, 's', color='black', label='True Values')
plt.xlabel('Predicted Log Opacity')
plt.ylabel('True Log Opacity')
plt.legend(loc='lower right')
plt.show()

#%%
''' Log Opacity - prediction vs truth '''
time = [x for x in range(1000)]

plt.style.use('seaborn-whitegrid')
plt.ylim([-7,2])
plt.title('Log Opacity (range -4 - -2): Predicted vs True Values')
plt.scatter(time, Lpred[0:1000], color='red', label='Predicted')            
plt.scatter(time, yL_test[0:1000], color='black', label='True Value')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend(loc='lower right')
plt.show()
#%%
''' Width Plot '''

plt.style.use('seaborn-whitegrid')
plt.title('CNN Regression of Width (range 0 - 5) with 41 Measurements')
plt.errorbar(Wpred, yW_test, yerr=rmseW, fmt='o', color='red',
             ecolor='cyan', elinewidth=3, capsize=0, label='Predicted Values with RMSE error');
#plt.plot(test_lo, test_w, 's', color='black', label='True Values')
plt.xlabel('Predicted Width')
plt.ylabel('True Width')
plt.legend(loc='lower right')
plt.show()

#%%
''' Width - prediction vs truth '''
time = [x for x in range(1000)]

plt.style.use('seaborn-whitegrid')
plt.ylim([3,14])
plt.title('Width (range 0 - 5): Predicted vs True Values')
plt.scatter(time, Wpred[0:1000], color='red', label='Predicted')            
plt.scatter(time, yW_test[0:1000], color='black', label='True Value')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend(loc='lower right')
plt.show()

#%%
''' Regressing the Log Opacity and using it to calculate FWHM'''

#Regressing Log Opacity
rmseL = 1
while (rmseL > 0.5):
    Lmodel = Sequential()
    Lmodel.add(Conv1D(32, 1, activation="relu", input_shape=(1,41)))
    Lmodel.add(Conv1D(32, 1, activation="relu", input_shape=(1,41)))
    Lmodel.add(Flatten())
    Lmodel.add(Dense(10, activation="relu"))
    Lmodel.add(Dense(1))
    Lmodel.compile(loss="mse", optimizer="adagrad")
    Lmodel.fit(X_train, yL_train, batch_size=12,epochs=2000, verbose=0)
     
    yLpred = Lmodel.predict(X_test)
     
    print(Lmodel.evaluate(X_train, yL_train))
    #Lmodel_weights = Lmodel.save_weights()
    rmseL = np.sqrt(metrics.mean_squared_error(yL_test, yLpred))
    print("Log Opacity MSE: %.4f" % mean_squared_error(yL_test, yLpred))
    print("Log Opacity RMSE: %.4f" % rmseL) 

Lmodel.save("D:/Radiometer/Gordons_project/Vary_Peak_Sigma/Full_Noise/Figures/CNN_model_JAN21_Log_Opacity.h5")
print("Saved model to disk")

#%%
''' Test-Train Split 2 '''
num=50000
train = num*.75
train = int(train)
test = num-train
X_test = Signal_new[train:num, :, :]
# Load model and apply to data
#mlL = load_model('D:/Radiometer/Gordons_project/Vary_Peak_Sigma/Full_Noise/Figures/CNN_model_Feb11_LogOpac_1_8756.h5') # loading the Log Opacity model

ypred = Lmodel.predict(Signal_new) # applying the model for Log Opacity to the dataset / want 50000 values?


#%%
'''
Polynomial regression 
Use for full signal instead of just Log Opacity and Width individually
'''

poly = PolynomialFeatures(degree=2)
X_2 = poly.fit_transform(X_train)
print(X_2.shape)
print(X_2[0])
model = Sequential([Dense(units=1, input_shape=[3])])
optimizer = Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss='mean_squared_error')
#tf_history = model.fit(X_2, y_scaled, epochs=500, verbose=True)
tf_history = model.fit(X_2, yL_test, epochs=10, verbose=True)
plt.plot(tf_history.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.title('Smartcomputerlab')
plt.show()
mse = tf_history.history['loss'][-1]
y_hat = model.predict(X_2)

#%%
'''*******Failed Attempts beyond this point********'''
#%%
''' Lorentz smoothing to help simplify the curve so the FWHM is easier to calculate
Code Source: https://docs.astropy.org/en/stable/convolution/kernels.html
This didn't inprove the results at all.
FWHM RMSE's still in the 1.7-1.9 range
'''
num=50000
train = num*.75
train = int(train)

gauss_kernel = Gaussian1DKernel(2)
SMData = np.zeros((num,41))
for i in range(num):
    data_gauss = convolve(Signal[i,:], gauss_kernel)
    SMData[i,:] = data_gauss

SMData = SMData.reshape(num, 1, 41)
X_trainLO = SMData[0:train, :, :]
X_testLO = SMData[train:num, :, :]
print(X_trainLO.shape, X_testLO.shape)

''' Plotting smoothed vs original curve '''

tobs = Signal_new[2,:]
tobs0 = tobs.reshape(41)

smoothed = SMData[2,:]
smoothed0 = smoothed.reshape(41)

fig, ax = plt.subplots()
ax.grid(True)
ax.set_ylim(950,1005)
ax.set_xlim(-10,10)
plt.subplots_adjust(left= .15, bottom=0.30)

freq = np.arange(-10.0, 10.5, 0.5) # Frequency range -10 to 10 MHz 
l, = plt.plot( freq, smoothed0, linestyle = ':', lw=3, color='blue', label='Smoothed data')
m, = plt.plot( freq, tobs0, linestyle = ':', lw=3, color='black', label='Original data')

plt.title('SSOLVE Observation vs Lorentz Smoothing', fontsize = 18)
plt.ylabel('T (K)', fontsize = 12)
plt.xlabel('Offset Freq (MHz)', fontsize = 10)
plt.legend(loc='upper right')
ax.margins(x=1)

plt.show()


'''RNN then CNN: Log Opacity
'''
Lmodel = Sequential()
Lmodel.add(LSTM((1), batch_input_shape=(None,1,41), return_sequences=True))
Lmodel.add(Conv1D(32, 1, activation="relu", input_shape=(1,41)))

Lmodel.add(Dense(50, activation='relu'))
Lmodel.add(Flatten())
Lmodel.add(Dense(1, activation='relu'))
Lmodel.compile(loss="mse", optimizer="adam")
Lmodel.fit(X_train, yL_train, batch_size=12,epochs=2000, verbose=0)
 
yLpred = Lmodel.predict(X_test)
 
print(Lmodel.evaluate(X_train, yL_train))
#Lmodel_weights = Lmodel.save_weights()
rmseL = np.sqrt(metrics.mean_squared_error(yL_test, yLpred))
print("Log Opacity MSE: %.4f" % mean_squared_error(yL_test, yLpred))
print("Log Opacity RMSE: %.4f" % rmseL) 


# save model and architecture to single file
Lmodel.save("D:/Radiometer/Gordons_project/Vary_Log_Width/Figures/RNN_CNN_model_JAN11_Log_Opacity.h5")
print("Saved model to disk")

''' RNN then CNN: Width '''

Wmodel = Sequential()
Wmodel.add(LSTM((1), batch_input_shape=(None,1,41), return_sequences=True))
Wmodel.add(Conv1D(32, 1, activation="relu", input_shape=(1,41)))
Wmodel.add(Flatten())
#Wmodel.add(Dense(10, activation="relu"))
Wmodel.add(Dense(1))
Wmodel.compile(loss="mse", optimizer="adam")
Wmodel.fit(X_train, yW_train, batch_size=12,epochs=2000, verbose=0) 
yWpred = Wmodel.predict(X_test)
 
print(Wmodel.evaluate(X_train, yW_train))
#Lmodel_weights = Lmodel.save_weights()
rmseW = np.sqrt(metrics.mean_squared_error(yW_test, yWpred))
print("Width MSE: %.4f" % mean_squared_error(yW_test, yWpred))
print("Width RMSE: %.4f" % rmseW)

# save model and architecture to single file
Wmodel.save("D:/Radiometer/Gordons_project/Vary_Log_Width/Figures/RNN_CNN_model_JAN11_Width.h5")
print("Saved model to disk")
















