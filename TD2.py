from __future__ import absolute_import , division , print_function , unicode_literals

# TensorFlow and tf . keras
import tensorflow as tf
from tensorflow import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import regularizers

# Helper libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
"""print(pd.DataFrame(train_data))
print(type(train_data)) #nump.ndarray
print(type(test_data))
print(type(train_labels))
print(type(test_labels))
print(train_data.shape, train_labels.shape, test_data.shape, test_labels.shape)

print(type(test_data[0]))
print(len(test_data[0]))
print(len(test_data[1]))
print(len(test_data[25000-1]))

print(type(train_data[0]))
print(len(train_data[0]))
print(len(train_data[1]))

print(len(train_data[25000-1]))
print(test_labels)"""

def vectorize_sequences( sequences , dimension =10000) :
    # Create an all - zero matrix of shape ( len ( sequences ) , dimension)
    results = np . zeros (( len ( sequences ) , dimension ) )
    for i , sequence in enumerate ( sequences ) :
        results [i , sequence ] = 1. # set specific indices ofresults [ i ] to 1 s
        
    return results

# Our vectorized training data
x_train = vectorize_sequences( train_data )
# Our vectorized test data
x_test = vectorize_sequences( test_data )
# Our vectorized labels
y_train = np.asarray( train_labels ).astype( "float32")
y_test = np.asarray( test_labels ).astype( "float32")

# Define the network model and its arguments. 
# Set the number of neurons/nodes for each layer:
model = keras.Sequential()
model.add(Dense(16, activation="relu" ,input_shape=(10000,), kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(Dense(16, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)  ))
model.add(Dense(1, activation="sigmoid" ))


# Compile the model and calculate its accuracy:
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc']) #binary_crossentropy for classification loss. rmsprop for declining gradients

# Print a summary of the Keras model:
#model.summary()
# Define the network model and its arguments. 
# Set the number of neurons/nodes for each layer:
smol_model = keras.Sequential()
smol_model.add(Dense(4, activation="relu" ,input_shape=(10000,), kernel_regularizer=keras.regularizers.l2(0.001)))
smol_model.add(Dense(4, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001) ))
smol_model.add(Dense(1, activation="sigmoid" ))


# Compile the model and calculate its accuracy:
smol_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc']) #binary_crossentropy for classification loss. rmsprop for declining gradients

# Print a summary of the Keras model:
#smol_model.summary()

#Fitting train data using the model using test data as validation
original_hist = model.fit(x_train, y_train, epochs=20, batch_size=512, validation_data=(x_test, y_test))

smol_original_hist = smol_model.fit(x_train, y_train, epochs=20, batch_size=512, validation_data=(x_test, y_test))

model_val_loss = original_hist.history["val_loss"]
smol_model_val_loss = smol_original_hist.history["val_loss"]

epochs = y=range(1,21)
plt.plot(epochs, model_val_loss, "b+", label="Original Model" )
plt.plot( epochs, smol_model_val_loss, "bo", label="Smoller Model")
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.legend()
plt.show()

# L1 regular ization
regularizers.l1(0.001)
# L1 and L2 regula rization at the same time
regularizers.l1_l2( l1 =0.001 , l2 =0.001)