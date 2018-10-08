import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils




def baseline_model(num_pixels,num_classes):
    model = Sequential()
    model.add(Dense(units = num_pixels,input_shape=(num_pixels,),kernel_initializer='normal',activation = 'relu'))
    model.add(Dense(units = num_classes,kernel_initializer='normal',activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model


def baseline_model2():
    model = Sequential()
    model.add(Conv2D(32,(5,5),input_shape=(28,28,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(units=64,activation='relu'))
    model.add(Dense(units=10,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model
def main():
    #preprocessing data
    seed = 7
    np.random.seed(seed)
    (X_train,y_train),(X_test,y_test)  = mnist.load_data()
    num_pixels = X_train.shape[1]*X_train.shape[2]
    X_train = X_train.reshape(X_train.shape[0],num_pixels).astype('float32')
    X_test = X_test.reshape(X_test.shape[0],num_pixels).astype('float32')
    X_train = X_train/255
    X_test = X_test/255
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = 10
    print('X_train shape: {0}\ny_train shape: {1}\nX_test shape {2}\ny_test shape:{3}\n'.format(X_train.shape,y_train.shape,X_test.shape,y_test.shape))
    
    #build model
    model = baseline_model(num_pixels,num_classes)
    #Fit the model
    model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=20,verbose=2)
    scores = model.evaluate(X_test,y_test,verbose=0)
    print(scores)


def main2():
    seed = 7
    np.random.seed(seed)
    (X_train,y_train),(X_test,y_test) = mnist.load_data()
    #reshape to be (samples,channels,width,height)
    X_train = X_train.reshape(X_train.shape[0],28,28,1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0],28,28,1).astype('float32')
    X_train = X_train/255
    X_test = X_test/255
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]
    print(f'main2:\nX_train shape:{X_train.shape}\nX_test shape:{X_test.shape}\ny_train shape:{y_train.shape}\ny_test shape:{y_test.shape}')
    
    #build convolutional network
    model = baseline_model2()
    model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=128,verbose=2)
    #Final evaluation of the model
    scores = model.evaluate(X_test,y_test,verbose=0)
    print(f'CNN accuracy:{scores[1]}')



if __name__ == "__main__":
    main2()
