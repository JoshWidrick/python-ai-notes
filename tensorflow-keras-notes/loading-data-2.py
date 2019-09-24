import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


# specify data directory
DATADIR = "C:/Users/joshu/OneDrive/Documents/GitHub/twovector/StockRelation/src/tensorflow-keras-notes/DataSets/PetImages"
# specify possible categories
CATEGORIES = ["Dog", "Cat"]
# image size for sizeXsize
IMG_SIZE = 64


"""
loading the data
---
we load in each image as an image array, greyscaled so that it is a 2d array instead of something more complicated.
we then need to normalize the sizes of the images
"""
# for category in CATEGORIES:
#     path = os.path.join(DATADIR, category)  # path to cats or dogs dir
#     for img in os.listdir(path):
#         img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GREYSCALE)  # convert to greyscale here
#
# IMG_SIZE = 64
# new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))


"""
creating training dataset
---
we use everything from above to load in the data while indexing it.
"""
training_data = []


def create_training_data():
    for category in CATEGORIES:
        path = DATADIR + "/" + category  # path to cats or dogs dir
        class_numb = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_path = path + "/" + img
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # convert to greyscale here
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_numb])
            except Exception as e:
                pass


create_training_data()


"""
balancing the data
---
balancing the training data is important to not have the loss calculated in a weighted way.
we do this by first shuffling the data.
then we separate it into x and y, the variables we will use right before we insert it into the nn.
in general a capital (X) is usually your feature set, and a lowercase (y) is your labels.
we then need to convert X into an array, and reshape it to be (-1) features(can be any #) with the shape 
(IMG_SIZExIMG_SIZE) by (1) (because it is greyscale. 
we then want to save the dataset so we don't have to remake the dataset every time. here we do it using pickle.
to open this data, we use pickle_in = open("X.pickle", "rb") and then X = pickle.load(pickle_in)
"""
import random

random.shuffle(training_data)

X = []
y = []
for features, label in training_data:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

import pickle

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
