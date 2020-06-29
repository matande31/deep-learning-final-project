import numpy as np
import cv2

import trainer

classes = ["5", "10", "25", "50", "100"]
model = None


def predict_single_image():
    print("Please input the image path:")
    path = input()
    image = cv2.imread(path)
    
    predictions = model.predict(np.array([image]))
    print(f"Single image class: {classes[np.argmax(predictions)]}c")


def predict_test_folder():
    images, labels = trainer.extract_images_and_labels("dataset/test")
    predictions = model.predict(np.array(images))
    success_counter = 0
    for i in range(len(predictions)):
        predicted_label = classes[np.argmax(predictions[i])]
        if predicted_label == labels[i]:
            success_counter += 1
            
    accuracy =  success_counter / len(images) * 100
    print(f"Accuracy: {accuracy}%")

