import tensorflow as tf
import sys
import pandas as pd
import numpy as np
import os
import random
import cv2
import glob
from pathlib import Path

# The input arguments for the VM
in_dir = sys.argv[2]
out_dir = sys.argv[4]

# get the data path from the above directory
data_path = os.path.join(in_dir, '*g')
img_folder = glob.glob(data_path)

# read the training data
df_profile = pd.read_csv('/Users/mattjosem/tcss455/training/profile/profile.csv')
# df_profile = pd.read_csv('{}profile/profile.csv')
profile = df_profile.loc[:, ['userid', 'gender']]

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

    # If no faces found, return None
    if len(f) == 0:
        return None

    # Find the coordinates of the face in the image
    (x, y, w, h) = f[0]

    return image[y:y+w, x:x+h]


print("Detecting faces in images, normalizing...")

# iterate over the images in the folder
for pic in img_folder:

    # Greyscale since images don't need to have color, in this
    # case at least, for the gender classifying
    img = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)

    face = detect_face(img)

    # Gets the user id from the file name
    image_id = Path(pic).stem

    # Checks if a face was found, if NONE found then doesn't add
    # to the img_data
    if face is not None:
        new_img = cv2.resize(face, (PIC_SIZE, PIC_SIZE))
        # appends the image and it's gender to the img_data
        img_data.append([new_img, image_id])

    # used to print to console the progress on the image normalization
    # after extracting faces and adding image/gender to img_data
    if image_iterator_countdown % 1000 == 0:
        print("%d images completed..." % image_iterator_countdown)
    image_iterator_countdown += 1

print("Pictures complete!")
print("SIZE OF DATA-SET AFTER COLLECTING FACES: ", len(img_data))

# Randomize the data
random.shuffle(img_data)

# Used to hold the images
test_X = []

# List of image ID's
id_list = []

# Appends the images and genders from shuffled
for image_arrays, image_id in img_data:
    test_X.append(image_arrays)
    id_list.append(image_id)

# make into numpy array
test_X = np.array(test_X).reshape(-1, PIC_SIZE, PIC_SIZE, 1)

# normalize inputs from 0-255 to 0-1
test_X = np.array(test_X / 255)
# train_y = np.array(train_y)

# load the model and look at the summary to confirm the accurate model
model = tf.keras.models.load_model('saved_image_model.h5')
model.summary()

print("Making Predictions...")

predicted_gender_arr = []

for pictures in test_X:

    picture = pictures.reshape(1, PIC_SIZE, PIC_SIZE, 1)

    prediction = model.predict(picture)
    predicted_gender = 0

    if prediction[0][0] > .5:
        predicted_gender = 1.0
    elif prediction[0][0] <= .5:
        predicted_gender = 0.0
    predicted_gender_arr.append(predicted_gender)

print("Creating XML's...")
# Iterate through each of the results in the lists and write them out to xml.
for idx in range(0, len(predicted_gender_arr)):
    the_id = id_list[idx]
    predicted_gender = predicted_gender_arr[idx]

    # Open a file for the user of the given userid and write out their results
    out_file = open('{}/{}.xml'.format(out_dir, the_id), 'w')

    out_string = '<user\n  \
                        id="{}"\n \
                        gender="{}"\n \
                />'.format(the_id, predicted_gender)

    out_file.write(out_string)
    out_file.close()
