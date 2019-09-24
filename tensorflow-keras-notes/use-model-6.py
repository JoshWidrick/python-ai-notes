import cv2
import tensorflow as tf


CATEGORIES = ["Dog", "Cat"]


def prepare(filepath):
    IMG_SIZE = 64
    im_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(im_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model('dog-v-cat-64x3-cnn.model')

prediction = model.predict([prepare('YOURPETIMAGE.jpg')])

print('cat' if prediction == 1 else 'dog')
