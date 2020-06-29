import os
import keras

import trainer
import tester


classes = ["5", "10", "25", "50", "100"]
model = None



def load_model():
    if not os.path.isfile("saved_model"):
        print("There is not saved model")
        return
    print("Loading model...")
    global model
    model = keras.models.load_model('saved_model')
    print("Done!")

def print_menu():
    print("Choose an option:")
    print("1. Load existing model")
    print("2. Train new model")
    print("3. Predict single image")
    print("4. Predict test folder")
    print("5. Exit")
    

def main():
    print("Welcome!")

    actions = [load_model, trainer.train_model, tester.predict_single_image, tester.predict_test_folder]

    loop = 1
    choice = 0
    while loop == 1:
        print_menu()
        choice = int(input()) - 1
        
        # check if to exit from app
        if choice == 4:
            loop = 0
            break
        
        # validate model is loaded for predictions 
        if model == None and choice > 1:
            print("You need to load/train model first")
            continue
        
        actions[choice]()

if __name__ == "__main__":
    main()
   
 

# coins -> 1) train data 2) test data
# train -> trained model
# predict(img) -> coin type
    
    
# 1 - Load model from file
# 2 - train model
# 3 - predice single image (path) -> class
# 4 - predict test folder -> accuracy 