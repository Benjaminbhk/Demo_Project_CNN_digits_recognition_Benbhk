from tensorflow.keras import datasets, layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import expand_dims
from tensorflow.keras.utils import to_categorical
import pickle as pkl
import numpy as np
from lib_func import fun, img_normalizer

class DigitRecognition():

    def __init__(self):

        # init the class and import the raw data

        print('Loading the data')
        (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data(path="mnist.npz")
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        print('Data loaded')


    def preproc(self):
        print('Process the data')
        # Process the X data to feet the model

        ## Maximize image contrast
        ## Greater performance with the hand drawing app

        vfunc = np.vectorize(fun)

        for i in range(0,len(self.X_train)):
            self.X_train[i] = vfunc(self.X_train[i])
            print(f'{round(i*100/len(self.X_train),1)} %', end = '\r')

        for i in range(0,len(self.X_test)):
            self.X_test[i] = vfunc(self.X_test[i])
            print(f'{round(i*100/len(self.X_test),1)} %', end = '\r')

        ## Normalisation of the data (0_255 -> -0.5_0.5)

        self.X_train = img_normalizer(self.X_train)
        self.X_test = img_normalizer(self.X_test)

        ## Inputs the RGB dimensionality in X_train and X_test

        self.X_train = expand_dims(self.X_train, axis=-1)
        self.X_test = expand_dims(self.X_test, axis=-1)

        # Process the y data to feet the model

        ## Categorize the y_train and y_test

        self.y_train = to_categorical(self.y_train, num_classes=10)
        self.y_test = to_categorical(self.y_test, num_classes=10)

        print('Process finish')

        return self



    def initialize_model(self):
        print('Creation of the model')
        model = Sequential()

        ### First convolution & max-pooling
        model.add(layers.Conv2D(8, (4,4), strides=(1,1), input_shape=(28, 28, 1), activation="relu", padding='same'))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))

        ### Second convolution & max-pooling
        model.add(layers.Conv2D(16, (3,3), strides=(1,1), activation="relu", padding='same'))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))

        ### Flattening
        model.add(layers.Flatten())

        ### One fully connected
        model.add(layers.Dense(10, activation='relu'))

        ### Last layer (let's say a classification with 10 output)
        model.add(layers.Dense(10, activation='softmax'))

        ### Model compilation
        model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

        self.model = model

        print('Model created')
        return self.model


    def fit(self,epochs=10,batch_size=4,verbose=1,patience=20,restore_best_weights=True):

        print('Fit the model')

        es = EarlyStopping(patience=patience,
                           restore_best_weights=restore_best_weights)

        self.model.fit(self.X_train,
                       self.y_train,
                       validation_split=0.3,
                       epochs=epochs,
                       batch_size=batch_size,
                       verbose=verbose,
                       callbacks=[es])

        return self.model

    def save_model(self,version='-Test'):
        pkl.dump(self.model, open(f'Demo_Project_CNN_digits_recognition_Benbhk/Models/numbers_recognition_model_V{version}', 'wb'))
        pass

if __name__ == "__main__" :

    digit_reco = DigitRecognition()
    digit_reco.preproc()
    digit_reco.initialize_model()
    digit_reco.fit(epochs=1, batch_size=32)
    digit_reco.save_model()
