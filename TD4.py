from keras.datasets import cifar10
(x_train,y_train), (x_test, y_test) = cifar10.load_data()

import numpy as np
x_train = np.float32(x_train/255.0)
x_test = np.float32(x_test/255.0)

import keras
model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape = (32, 32, 3)),#32 filters
    keras.layers.MaxPooling2D(pool_size= (2,2), strides=2),
    keras.layers.Conv2D(64, kernel_size=3, activation='relu'),
    keras.layers.MaxPooling2D(pool_size = (2,2), strides= 2),
    keras.layers.Conv2D(64, kernel_size= 3, activation = 'relu'),
    keras.layers.MaxPooling2D(pool_size = (2,2), strides= 2),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation = 'relu'),
    keras.layers.Dense(10, activation = 'softmax')
])
model.summary()

model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size = 128, epochs=20, validation_split=0.2)

import matplotlib.pyplot as plt

predictions = model.predict(x_test)
predictions[0]

print(np.argmax(predictions[0]))
print(y_test[0])

plt.figure()
plt.imshow(x_test[0])
plt.colorbar()
plt.grid(False)
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy (fractional')
plt.legend(['training accuracy', 'validation accuracy'], loc='best')

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy (fractional')
plt.legend(['training loss', 'validation loss'], loc='best')

"""The provided code performs image classification using the CIFAR-10 dataset using a convolutional neural network (CNN) implemented with Keras.

Here is a breakdown of the code:

    Importing the required libraries and loading the CIFAR-10 dataset:
        The code begins by importing the cifar10 module from Keras to access the CIFAR-10 dataset.
        It then loads the dataset, splitting it into training and testing sets, assigning them to (x_train, y_train) and (x_test, y_test) respectively.

    Preprocessing the data:
        The numpy library is imported to handle numerical operations.
        The pixel values in the training and testing images are scaled between 0 and 1 by dividing them by 255.0. This normalization step is performed to ensure that the values are in a suitable range for the neural network.

    Creating the CNN model:
        The keras module is imported to access the Keras API for building deep learning models.
        The code defines a sequential model, which is a linear stack of layers.
        The model architecture consists of multiple convolutional layers with rectified linear unit (ReLU) activation and max pooling layers.
        The final layers include flattening the input, adding fully connected (dense) layers with ReLU activation, and a final dense layer with softmax activation (for multi-class classification).

    Compiling and training the model:
        The model is compiled using the Adam optimizer and sparse categorical cross-entropy loss.
        The fit function is called to train the model on the training data for a specified number of epochs (20 in this case).
        During training, a validation split of 0.2 is used to evaluate the model's performance on a portion of the training data.

    Making predictions and visualizing results:
        The model is used to predict the classes of the test images, storing the predictions in the predictions variable.
        The first prediction is printed using np.argmax(predictions[0]).
        The actual label of the first test image is printed using y_test[0].
        Matplotlib is used to display the first test image, along with a color bar and grid lines.
        Two plots are generated using the history object, showing the training and validation accuracy as well as the training and validation loss over the epochs.

Overall, this code demonstrates the process of building and training a CNN model for image classification using the CIFAR-10 dataset, along with visualizing the results.
"""