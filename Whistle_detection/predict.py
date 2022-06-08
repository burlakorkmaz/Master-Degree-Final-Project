#********************* IMPORTS

import tensorflow as tf 
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from os import listdir
from os.path import isfile, join
import cv2
import shutil
import pandas as pd


#%%

#********************* FUNCTIONS

def predict(model, images_path, file_path):
    image = cv2.imread(join(images_path, file_path))
    image = cv2.resize(image, (224, 224))
    image = np.resize(image,(1,224,224,3))
    image = preprocess_input(image)
    prediction = model.predict(image)
    
    return prediction

def move_file(images_path, file_path, dst_dir):
    src_dir = join(images_path, file_path)
    shutil.move(src_dir,dst_dir)

def prepare_csv_data(file_path, record_names, positive_initial, positive_finish):
    part = file_path.split('c-')

    name = part[0] + "c"
    record_names.append(name)
    
    ini = part[1].replace(".jpg", "")
    ini = float(ini)
    positive_initial.append(ini)
    
    fin = ini + 0.8
    fin = round(fin, 1)
    positive_finish.append(fin)
    
    return record_names, positive_initial, positive_finish

def save_csv(record_names, positive_initial, positive_finish, class_1_scores, csv_path):
    df = {'file_name': record_names,
    'initial_point': positive_initial,
    'finish_point': positive_finish,
    'confidence': class_1_scores}

    df = pd.DataFrame(df)
    
    df.to_csv(csv_path, index=False)

#%%

#********************* MAIN

model_path = "models\\model_vgg.h5"
images_path = "images\\"
positive_dir = "predicted_images\\positive"
negative_dir = "predicted_images\\negative"
csv_path = "whistles.csv"

# the model
model = tf.keras.models.load_model(model_path)

# paths of all files
all_files_path = [f for f in listdir(images_path) if isfile(join(images_path, f))]

# lists to store data
record_names = []
positive_initial = []
positive_finish = []
class_1_scores = []

# all predictions results
predictions = []

# reading file paths 1 by 1
for file_path in all_files_path:
    
    # prediction on the given image
    prediction = predict(model, images_path, file_path)
    predictions.append([file_path, prediction])
    
    # if the class 1 has higher confidence than class 0
    if (prediction[0][1] > prediction[0][0]):
        
        # carry the positive image to its folder.
        move_file(images_path, file_path, positive_dir)
        
        # storing the positive images confidences
        class_1_scores.append(prediction[0][1])
        
        # preparing arrays for the csv
        record_names, positive_initial, positive_finish = prepare_csv_data(file_path,
                                                                           record_names,
                                                                           positive_initial,
                                                                           positive_finish)
    else:
        
        # carry the negative image to its folder.
        move_file(images_path, file_path, negative_dir)
        
    

#saving the csv
save_csv(record_names, positive_initial, positive_finish, class_1_scores, csv_path)
















