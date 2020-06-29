from keras import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout

def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(100, 100, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(data_format="channels_last", pool_size=(2, 2)))
    
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(data_format="channels_last", pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(data_format="channels_last", pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.75))          # 0.5
    model.add(Dense(5))               # 5 is the number of classes
    model.add(Activation('softmax'))
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',              # adadelta
        metrics=['accuracy']
    )
    
    return model