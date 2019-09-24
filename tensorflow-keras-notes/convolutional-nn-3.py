"""
CNN Overview
---
a convolutional neural net's structure usually follows:
`Convolution -> Pooling -> Convolution -> Pooling -> ... -> Fully Connected Layer -> Output`
Convolution is used to find useful things, Pooling is used to pool these useful things together.
in the case of images, this process is done through selecting a window on the image, assigning it a value,
and then sliding the window and repeating. this gives us a bunch of convolutions, which we then pool with windows,
usually through max pooling. all of this leads to a process that slowly extracts data from the images.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D


# load in the data
import pickle

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))


"""
normalizing the data, scaling the data
---
we usually want to normalize the data. here we are doing that by scaling it.
"""
X = X/255.0

model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(Activation("relu"))  # could add pooling layer here, however this is where we are putting the activation layer in this model
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# flatten the data for the dense layer
model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

# output layer
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.fit(X, y, batch_size=32, epochs=3, validation_split=0.1)
