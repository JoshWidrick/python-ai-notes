from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

# from convolutional-nn-3.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle


"""
name the model
---
start by making a dynamic name to give to the model each time you train it. this helps separate the models as you train
them.
"""
NAME = 'cats-vs-dog-cnn-64x2-{}'.format(int(time.time()))


"""
setup tensorboard
---
run `tensorboard --logdir=logs/` from cmd in this dir to get webpanel up.
"""
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))


"""
processing options
---
"""
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gup_options=gpu_options))


# from convolutional-nn-3.py
X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

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

""" ADD tensorboard to the callbacks """
model.fit(X, y, batch_size=32, epochs=3, validation_split=0.1, callbacks=[tensorboard])
