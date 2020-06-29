import random
import shutil
import os

import numpy as np
import cv2
from keras.utils import to_categorical 

from model import create_model


classes = ["5", "10", "25", "50", "100"]
model = None

def train_model():
    all_coins_path = "dataset/all_coins"
    
    # check if test folder already exist - if not its means we need to prepare the data
    if not os.path.isdir("dataset/test"):
        prepare_data(all_coins_path)
        
    train_images, train_labels = extract_images_and_labels("dataset/train")
    train_labels_data = prepare_labels(train_labels)
    train(train_images, train_labels_data)
    save_model()
    
def copy_files_into_folder(files_array, output_path, original_path):
    for file in files_array:
        shutil.copyfile(f"{original_path}/{file}", f"{output_path}/{file}")


def prepare_data(all_coins_path):
    files = []
    
    for r, d, f in os.walk(all_coins_path):
        for file in f:
            if '.jpg':
                files.append(file)
    
    random.shuffle(files)
    
    test_data = files[:int(len(files) * 0.1)]
    train_data = files[int(len(files) * 0.1):]
    
    os.mkdir("dataset/test")
    os.mkdir("dataset/train")
    
    copy_files_into_folder(test_data, "dataset/test", all_coins_path)
    copy_files_into_folder(train_data, "dataset/train", all_coins_path)
    
def train(train_images, labels):
    global model
    model = create_model()
    model.fit(
        x=np.array(train_images),
        y=labels,
        epochs=20,
        validation_split=0.15,
        batch_size=500,
        verbose=1        
    )
    
def extract_images_and_labels(folder_path):
    files = []
    images = []
    labels = []
    for r, d, f in os.walk(folder_path):
        for file in f:
            if '.jpg':
                files.append(file)
                
    random.shuffle(files)
                
    for file in files:
        image = cv2.imread(f"{folder_path}/{file}")
        images.append(image)
        label = file.split('_')[0]
        labels.append(label)          
    return images, labels

def prepare_labels(labels):    

    dict_labels = {}
    for i in range(len(classes)):
        dict_labels[classes[i]] = i

    labels_indexes = []
    for label in labels:
        labels_indexes.append(dict_labels[label])
        
    return to_categorical(labels_indexes)
        
def save_model():
    model.save("saved_model")
