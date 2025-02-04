from __future__ import absolute_import , division , print_function , unicode_literals

# TensorFlow and tf . keras
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.mnist
mnist_data = data.load_data()

( train_images , train_labels ) ,( test_images , test_labels ) = mnist_data

print ( train_images[0])
print ( train_images[0].shape )



class_names = [ 'airplane' , 'automobile' , 'bird' , 'cat' , 'deer' , 'dog' , 'frog' , 'horse' , 'ship' , 'truck'] 
plt.figure(figsize=(20,20)) 
for i in range(25): 
  plt.subplot(5,5,i+1) 
  plt.xticks([]) 
  plt.yticks([]) 
  plt.grid(False) 
  plt.imshow(train_images[i]) 
  #, cmap=plt.cm.binary) 
  #plt.xlabel(class_names[train_labels[i][0]]) 
plt.show()

train_images = train_images / 255.0

model = keras.Sequential ([keras.layers.Flatten( input_shape =(28 , 28 , 1) ) , keras.layers.Dense(128 , activation = "relu") , keras.layers.Dense(10 , activation = "softmax") ])
