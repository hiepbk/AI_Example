# -*- coding: utf-8 -*-
#%% import
import numpy as np
from keras import Sequential, models
from keras.layers import Dense
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model


#%% load data
dataset = loadtxt('dulieu.csv',delimiter=',')

X = dataset[:,0:8]
y = dataset[:,8]
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size= 0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size = 0.2)

model = Sequential()
model.add(Dense(32,input_dim=8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary()

#%% Compile
model.compile(optimizer='adam',loss='binary_crossentropy',metrics =['accuracy'])
model.fit(X_train, y_train, batch_size=8,epochs=10,validation_data=(X_val,y_val))
#%% save model
model.save("mymodel.h5")


#%% load model without training

model_loaded = models.load_model('mymodel.h5')
# loss, acc = model_loaded.evaluate(X_test, y_test)
y_predict =  model_loaded.predict(X_test)
#%%
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)