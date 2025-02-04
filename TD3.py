import pandas as pd

wine = pd.read_csv('winequality-white.csv', sep=";")

#print(wine.head(50))

import matplotlib.pyplot as plt
plt.scatter(wine['fixed acidity'] ,wine['alcohol'] ,c = wine ['quality'] ,cmap = 'viridis')
plt.xlabel("fixed acidity")
plt.ylabel("alcohol")
#plt.show()
#3
from keras.utils import to_categorical
X = wine.iloc[:,range(11)]
print(type(X))
y = wine.iloc[:,[11]]
X = X.values
print(type(X))
y = y.values

y = to_categorical(y)

print(y.shape)
print(y[0])
#4
n_train = int(0.8*X.shape[0])
trainX , testX = X [: n_train , :] , X [ n_train : , :]
trainy , testy = y [: n_train ] , y [ n_train :]
#5
from tensorflow import keras

model1 = keras.Sequential([
    keras.layers.Dense(50, input_dim=11, activation='relu', kernel_initializer='he_uniform'),
    keras.layers.Dense(10, activation='softmax')
])

model1.summary()
#6
from keras.optimizers import SGD
lrate = 0.01
model1.compile(loss ='categorical_crossentropy', optimizer=keras.optimizers.SGD(learning_rate=lrate) ,metrics=['accuracy'])
#7
history = model1.fit(trainX, trainy, validation_data=(testX ,testy) ,epochs=200 , verbose=0)
#8
plt.figure()
plt.plot(history.history['accuracy'], label='train', color='r')
plt.plot(history.history['val_accuracy'], label='test', color='b')
plt.title('lrate='+ str(lrate) ,pad=-50)

plt.show()
#9
def fit_model(trainX, trainy, testX, testy, lrate):#fit a model and plot its performance

    model = keras.Sequential([
        keras.layers.Dense(50, input_dim=11, activation='relu', kernel_initializer='he_uniform'),
        keras.layers.Dense(10, activation = 'softmax')
    ])
    model.compile(loss ='categorical_crossentropy', optimizer = keras.optimizers.SGD(learning_rate = lrate),metrics=['accuracy'])

    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=200, verbose=0)

    plt.plot(history.history ['accuracy'], label ='train', color = 'r')
    plt.plot(history . history ['val_accuracy'], label ='test', color = 'b')
    plt.title('lrate ='+ str (lrate), pad = -50)

    plt.show()
#10
learning_rates = [1E-0, 1E-1, 1E-2, 1E-3, 1E-4, 1E-5, 1E-6, 1E-7]
for i in range(len(learning_rates)):
  plot_no=420 + (i +1)
  plt.subplot(plot_no)
  fit_model(trainX, trainy, testX, testy, learning_rates[i])
  plt.show()
#11
def fit_model2(trainX, trainy, testX, testy, momentum):#fit a model and plot its performance

    model = keras.Sequential([
        keras.layers.Dense(50, input_dim=11, activation='relu', kernel_initializer='he_uniform'),
        keras.layers.Dense(10, activation = 'softmax')
    ])
    model.compile(loss ='categorical_crossentropy', optimizer = keras.optimizers.SGD(learning_rate = 1E-05, momentum = momentum),metrics=['accuracy'])

    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=200, verbose=0)

    plt.plot(history.history ['accuracy'], label ='train', color = 'r')
    plt.plot(history . history ['val_accuracy'], label ='test', color = 'b')
    plt.title('momentum ='+ str (momentum), pad = -80)
#12
momentums = [0.0, 0.5, 0.9, 0.99]
for i in range(len(momentums)):
  plot_no = 220 + (i + 1)
  plt.subplot(plot_no)
  fit_model(trainX,trainy,testX,testy, momentums[i])
  plt.legend()
  plt.show()
#13
def decay_lrate(initial_lrate, decay, iteration):
      return initial_lrate * (1.0/(1.0 + decay * iteration))
decays = [1E-1 , 1E-2 , 1E-3 , 1E-4]
lrate = 0.01
n_updates = 200
for decay in decays :
  #calculate learning rates for updates
  lrates=[decay_lrate(lrate, decay, i)for i in range(n_updates)]
  #plot result
  plt.plot(lrates, label = str(decay))
plt.legend()
plt.show() 
#14
def fit_model3(trainX, trainy, testX, testy, decay):
      #fit a model and plot its performance

    model = keras.Sequential([
        keras.layers.Dense(50, input_dim=11, activation='relu', kernel_initializer='he_uniform'),
        keras.layers.Dense(10, activation = 'softmax')
    ])
    model.compile(loss ='categorical_crossentropy', optimizer = keras.optimizers.SGD(learning_rate = 1E-05, decay = decay),metrics=['accuracy'])

    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=200, verbose=0)

    plt.plot(history.history ['accuracy'], label ='train', color = 'r')
    plt.plot(history . history ['val_accuracy'], label ='test', color = 'b')
    plt.title('decay ='+ str (decay), pad = -70)

decay = [1E-1, 1E-2, 1E-3, 1E-4]
for i in range(len(decay)):
  plot_no = 220 + (i+1)
  plt.subplot(plot_no)
  fit_model3(trainX, trainy, testX, testy, decay[i])
  plt.legend()
  plt.show()
#15
# learning rate decay
def methods(methods, decay, iteration):
	return methods * (1.0 / (1.0 + decay * iteration))
#16
# fit a model and plot learning curve
def fit_model4(trainX, trainy, testX, testy, optimizer):
  # define model
  model = keras.Sequential([
                            keras.layers.Dense(50, input_dim=11, activation='relu', kernel_initializer='he_uniform'),
                            keras.layers.Dense(10, activation='softmax')
                            ])
  # compile model
  model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
  # fit model
  history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=200, verbose=0)
  # plot learning curves
  plt.plot(history.history['accuracy'], label='train')
  plt.plot(history.history['val_accuracy'], label='test')
  plt.title('optimizer='+str(optimizer), pad=-80)

#17
methods = ['sgd', 'rmsprop', 'adagrad', 'adam']
for i in range(len(methods)):
  # determine the plot number
  plot_no = 220 + (i+1)
  plt.subplot(plot_no)
  # fit model and plot learning curves for a decay rate
  fit_model4(trainX, trainy, testX, testy, methods[i])
  # show learning curves
  plt.legend()
  plt.show()
  
  """
  The provided code focuses on training a neural network model for wine quality classification using various optimization techniques.

Here is a breakdown of the code:

1. Importing the necessary libraries and loading the wine quality dataset:
   - The code starts by importing the `pandas` library as `pd`.
   - The wine quality dataset is read from a CSV file named 'winequality-white.csv' using `pd.read_csv()`. The data is separated by semicolons.
   - The code also imports the `matplotlib.pyplot` library for data visualization.

2. Visualizing the dataset:
   - A scatter plot is created using `plt.scatter()` to display the relationship between the "fixed acidity" and "alcohol" features of the wine dataset. The color of each data point represents the quality of the wine.
   - Labels are added to the x-axis and y-axis using `plt.xlabel()` and `plt.ylabel()` respectively.
   - The plot is commented out with `#plt.show()` to prevent it from being displayed immediately.

3. Preprocessing the dataset:
   - The independent features (X) are extracted from the dataset using `wine.iloc[:, range(11)]`.
   - The dependent variable (y), which represents wine quality, is extracted using `wine.iloc[:, [11]]`.
   - The X and y dataframes are converted to NumPy arrays using `.values` for further processing.
   - The y variable is one-hot encoded using `to_categorical()` from `keras.utils`.

4. Splitting the dataset into training and testing sets:
   - The training set size is determined by multiplying the total number of samples (X.shape[0]) by 0.8 and converting it to an integer using `int()`.
   - The X and y arrays are split into training and testing sets using array slicing.

5. Creating and summarizing the neural network model:
   - A sequential model is defined using `keras.Sequential()`.
   - The model architecture consists of two dense (fully connected) layers with ReLU activation.
   - The summary of the model is printed using `model1.summary()`.

6. Compiling the model:
   - The model is compiled using the stochastic gradient descent (SGD) optimizer with a specified learning rate (`lrate`).
   - Categorical cross-entropy is used as the loss function, and accuracy is used as the metric to evaluate the model's performance.

7. Training the model:
   - The model is trained using the `fit()` function on the training data, with the testing data used for validation.
   - The training is performed for 200 epochs with `history` used to store the training metrics.

8. Visualizing the training and validation accuracy:
   - A new figure is created using `plt.figure()`.
   - Two plots are generated to display the training and validation accuracy over the epochs using `plt.plot()`.
   - The title of the plot is set as the learning rate (`lrate`).
   - The plot is displayed using `plt.show()`.

9. Defining a function to fit a model and plot its performance:
   - The `fit_model()` function takes the training and testing data, as well as a learning rate (`lrate`) as parameters.
   - Inside the function, a new model is defined and compiled similarly to the previous model.
   - The model is trained and the training and validation accuracy over the epochs are plotted using `plt.plot()`.
   - The title of the plot is set as the learning rate (`lrate`).
   - The plot is displayed using `plt.show()`.

10. Looping over different learning rates and plotting the performance:
   - A list of learning rates is defined.
   - A loop iterates over the learning rates and

 calls the `fit_model()` function for each learning rate.
   - A subplot is created for each learning rate using `plt.subplot()`.
   - The function `fit_model()` is called within the loop to fit the model and plot its performance.
   - The legend and plot are displayed using `plt.legend()` and `plt.show()`.

11. Defining a function to fit a model with momentum and plot its performance:
   - The `fit_model2()` function takes the training and testing data, as well as a momentum value as parameters.
   - Inside the function, a new model is defined and compiled similarly to the previous models.
   - The model is trained and the training and validation accuracy over the epochs are plotted using `plt.plot()`.
   - The title of the plot is set as the momentum value.
   - The plot is displayed using `plt.show()`.

12. Looping over different momentum values and plotting the performance:
   - A list of momentum values is defined.
   - A loop iterates over the momentum values and calls the `fit_model2()` function for each momentum value.
   - A subplot is created for each momentum value using `plt.subplot()`.
   - The function `fit_model2()` is called within the loop to fit the model and plot its performance.
   - The legend and plot are displayed using `plt.legend()` and `plt.show()`.

13. Defining a function to calculate the learning rate decay:
   - The `decay_lrate()` function takes the initial learning rate, decay rate, and iteration as parameters.
   - It returns the updated learning rate based on the decay formula.

14. Looping over different decay rates and plotting the performance:
   - A list of decay rates is defined.
   - A loop iterates over the decay rates and calculates the learning rates for each update using `decay_lrate()`.
   - The learning rates are plotted against the number of updates using `plt.plot()`.
   - The legend and plot are displayed using `plt.legend()` and `plt.show()`.

15. Defining the `methods()` function to calculate learning rates based on different methods:
   - The `methods()` function takes the methods, decay rate, and iteration as parameters.
   - It returns the learning rate calculated using a specific method.

16. Defining the `fit_model4()` function to fit a model with a given optimizer and plot its performance:
   - The `fit_model4()` function takes the training and testing data, as well as an optimizer as parameters.
   - Inside the function, a new model is defined and compiled similarly to the previous models.
   - The model is trained and the training and validation accuracy over the epochs are plotted using `plt.plot()`.
   - The title of the plot is set as the optimizer name.
   - The plot is displayed using `plt.show()`.

17. Looping over different optimization methods and plotting the performance:
   - A list of optimization methods is defined.
   - A loop iterates over the methods and calls the `fit_model4()` function for each optimizer.
   - A subplot is created for each optimizer using `plt.subplot()`.
   - The function `fit_model4()` is called within the loop to fit the model and plot its performance.
   - The legend and plot are displayed using `plt.legend()` and `plt.show()`.

In summary, the code explores different optimization techniques such as learning rate variation, momentum, learning rate decay, 
and optimization methods to train a neural network model for wine quality classification. It visualizes the performance of the models using accuracy plots.
  """