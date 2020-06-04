
import pandas as pd
import os
import shutil

# Creating the data for the positive samples

FILE_PATH = "F:\\Covid CNN\\covid-chestxray-dataset-master\\covid-chestxray-dataset-master\\metadata.csv"
IMAGES_PATH = "F:\\Covid CNN\\covid-chestxray-dataset-master\\covid-chestxray-dataset-master\\images"

df = pd.read_csv(FILE_PATH)
print(df.shape)

df.head()
# findings column denotes the diagnosis

Target_dir = "F:\\Covid CNN\\Dataset\\Covid"

if not os.path.exists(Target_dir):
    os.mkdir(Target_dir)
    print("Covid folder created")


cnt = 0
# We need to filter out some X_rays which have back view/topview
# Therefore count only those xrays that have PA view (front view)


# We have copied 144 images of PA view to a new Target directory of Covid Patients
for (i,row) in df.iterrows():
    if row["finding"] == "COVID-19" and row["view"] == "PA":
        filename= row["filename"]
        image_path = os.path.join(IMAGES_PATH, filename)
        image_copy_path = os.path.join(Target_dir,filename)
        shutil.copy2(image_path, image_copy_path)
        print("Moving images")
        cnt += 1
        
print(cnt)

# Sampling of Kaggle dataset 50-50 images

import random
Kaggle_file_path = "F:\\Covid CNN\\chest-xray-pneumonia\\chest_xray\\train\\NORMAL"
Target_normal_dir = "F:\\Covid CNN\Dataset\\Normal"

image_names = os.listdir(Kaggle_file_path)

random.shuffle(image_names)

# Randomly shuffled the images and picking the first 144
# images of normal patients (training)
for i in range(142):
    image_name = image_names[i]
    image_path = os.path.join(Kaggle_file_path, image_name)
    
    target_path = os.path.join(Target_normal_dir, image_name)
    shutil.copy2(image_path, target_path)
    print("Copying image", i)
    


# Building Model

import numpy
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image

# CNN model in keras


model = Sequential()
model.add(Conv2D(32, kernel_size = (3,3), activation = "relu", input_shape = (224, 224, 3)))
model.add(Conv2D(64, (3,3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
