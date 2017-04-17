import numpy
from keras.datasets import cifar10
from keras.layers import Dense
from keras.constraints import maxnorm
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
K.set_image_dim_ordering('th')
numpy.random.seed(8)  # for reproducibility
# loading data
(P_train, Q_train), (P_test, Q_test) = cifar10.load_data()
P_train = P_train.astype('float32')
P_test = P_test.astype('float32')
P_train = P_train / 255.0         #The pixel values are in the range of 0 to 255 for each of the red, green and blue channels.
P_test = P_test / 255.0       # # diving data pixel by pixel 
Q_train = np_utils.to_categorical(Q_train)
Q_test = np_utils.to_categorical(Q_test)  # one hot encoding to transform the classes into a binary matrix
num_classes = Q_test.shape[1]
#neural network model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))  
#32 features are  mapped  with a size of 3×3, a rectifier activation function and a weight constraint of max norm set to 3
model.add(Dropout(0.2))
# dropout is set to 20% , for avoiding redundancy
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
#Max Pool layer with size 2×2
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
#Fully connected layer with 512 units and a rectifier activation function.
#Dropout is set to 50%
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))  # output layer
#Fully connected output layer with 10 units and a softmax activation function.
# Compile model

decay = 0.01/100
sgd = SGD(lr=0.01, momentum=0.8, decay=decay, nesterov=False)
# compiling model
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#print(model.summary())
history=model.fit(P_train, Q_train, validation_data=(P_test, Q_test), epochs=100, batch_size=32)
# Final evaluation of the model
scores = model.evaluate(P_test, Q_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
print(history.history.keys())
# import matplotlib.pyplot for plotting curves
import matplotlib.pyplot as pl  
# loss vs epoch
pl.plot(history.history['loss'])
pl.plot(history.history['val_loss'])
pl.title('model loss')
pl.ylabel('loss')
pl.xlabel('epoch')
pl.legend(['train', 'test'], loc='upper left')
pl.show()
# accuracy vs epoch
pl.plot(history.history['acc'])
pl.plot(history.history['val_acc'])
pl.title('model accuracy')
pl.ylabel('accuracy')
pl.xlabel('epoch')
pl.legend(['train', 'test'], loc='upper left')
pl.show()

# saving model
import json
saved_model = model.to_json()
with open('Image100.json', 'w') as outfile:
  json.dump(saved_model, outfile)

model.save_weights('Image100.h5')
