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
import json
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
from keras.models import model_from_json 
print('loading model')  
# copy the path of model    
with open('/home/iitp/Desktop/MODEL_OF_IMAGE/models/MODEL_OF_100 EPOCHS/IMAGE100.json', 'r') as architecture_file:
  model_architecture = json.load(architecture_file)


 
loaded_model = model_from_json(model_architecture)
#loading model
# copy the path of saved model
loaded_model.load_weights('/home/iitp/Desktop/MODEL_OF_IMAGE/models/MODEL_OF_100 EPOCHS/IMAGE100.h5')
epochs = 35
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=0.01, momentum=0.9, decay=decay, nesterov=False)
# compiling model
loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# evaluating model
print('evaluating')
scores =loaded_model.evaluate(P_test, Q_test, verbose=0)

print("Accuracy: %.2f%%" % (scores[1]*100))


