"""
TCSS 455
Matthew Molina
May 12th, 2020
Gender classification based on images as the sources
"""

import pandas as pd
import numpy as np
import os
import random
import cv2
import glob
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D

img_dir = "/Users/mattjosem/tcss455/training/image"  # Enter Directory of all images

# get the data path from the above directory
data_path = os.path.join(img_dir, '*g')
img_folder = glob.glob(data_path)

# read the training data
df_profile = pd.read_csv('/Users/mattjosem/tcss455/training/profile/profile.csv')
profile = df_profile.loc[:, ['userid', 'gender']]

# Dictionary that holds userid as key, value is gender
id_gender_dict = {}

for users, gens in profile.iterrows():
    id_gender_dict.update({gens['userid']: gens['gender']})

# Constant to store the width and height of images
PIC_SIZE = 100

# will hold resized, greyscaled images, as well as gender.
img_data = []

# counter to show progress of image processing
image_iterator_countdown = 0


# Used to detect the image for faces, returns face if found
def detect_face(image):
    # This is the classifier to detect faces, less accurate, but much quicker than Haar
    face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
    f = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)

    # If no faces found return None, prevents training on anything not a face
    if len(f) == 0:
        return None

    # Find the coordinates of the face in the image
    (x, y, w, h) = f[0]

    return image[y:y + w, x:x + h]


print("Beginning processing photos...")

# iterate over the images in the folder
for pic in img_folder:

    # Greyscale since images don't need to have color, in this
    # case at least, for the gender classifying
    img = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)

    # Extracts the face from the image if any
    face = detect_face(img)

    # Gets the user id from the file name
    user_id = Path(pic).stem

    # Checks if a face was found, if NONE found then doesn't add
    # to the img_data
    if face is not None:
        new_img = cv2.resize(face, (PIC_SIZE, PIC_SIZE))
        # appends the image and it's gender to the img_data
        img_data.append([new_img, id_gender_dict.get(user_id)])

    # used to print to console the progress on the image normalization
    # after extracting faces and adding image/gender to img_data
    if image_iterator_countdown % 1000 == 0:
        print("%d images completed..." % image_iterator_countdown)
    image_iterator_countdown += 1

# Randomize the data
random.shuffle(img_data)

# Used to hold the images
train_X = []
# Used to hold the genders
train_y = []

# Appends the images and genders from shuffled
for image_arrays, genders in img_data:
    train_X.append(image_arrays)
    train_y.append(genders)

# make into numpy array
train_X = np.array(train_X).reshape(-1, PIC_SIZE, PIC_SIZE, 1)

# normalize inputs from 0-255 to 0-1
train_X = np.array(train_X / 255)
train_y = np.array(train_y)


# Function creates the model
def create_model():
    model = Sequential()

    model.add(Conv2D(128, (4, 4), input_shape=train_X.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (4, 4)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (4, 4)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.1))
    model.add(Flatten())

    model.add(Dense(1, activation='sigmoid'))

    # Use binary since the output is either 1.0 or 0.0, for male and female
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# build the model
model = create_model()

# Fit the model
data = model.fit(train_X, train_y, validation_split=0.1, epochs=5, batch_size=32, verbose=1)
model.save('saved_image_model.h5')


# These are utilized to show the loss curves and accuracy curves
plt.plot(np.arange(0, 5), data.history['loss'])
plt.title('Training Loss')
plt.show()

plt.plot(data.history['accuracy'])
plt.plot(data.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(data.history['loss'])
plt.plot(data.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
