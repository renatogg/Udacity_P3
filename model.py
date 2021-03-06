from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.layers import Input
from keras.models import Sequential, Model
import cv2, numpy as np
from TrainingData import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from keras.applications.vgg16 import VGG16

center = 0
left = 1
right = 2
steering = 3
angleTS = 0.25

def vgg(inputshape):
    #Creates tensor to receive input with desired shape
    input_layer = Input(shape=inputshape)
    #Creates VGG16 base model with weights from imagenet. 
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_layer)
    #Extend base model to include two fully connected with dropout and a final 1 neuron layer for the steering.
    layer = base_model.output
    layer = Flatten()(layer) 
    layer = Dense(1024, activation='relu', name='fc')(layer)
    layer = Dropout(0.5)(layer)
    #The idea for this two neurons layers is that it gives a linear combination on how much to steer left, and how much to steer right.
    layer = Dense(2, activation='linear', name='fc2')(layer)    
    #Final layer, this is the steering output.
    layer = Dense(1, activation='linear', name='predictions')(layer)

    model = Model(input=base_model.input, output=layer)
    return model

## Load data from CSV file
def loadcsv(name,discard_zero):
    csvfile = open(name, 'rt')
    #Load all lines from csv file and closes it
    lines = csvfile.readlines()    
    csvfile.close()
    X = []
    y = []
    X_out = []
    y_out = []
    for i in range(len(lines)):        
        row = lines[i].split(', ')
        cimg = row[center]       
        limg = row[left]           
        rimg = row[right]        
        
        steer = float(row[steering])
        #Discard any steering below 5 degrees
        if discard_zero and abs(steer) < 2e-1:
            continue
        X.append(cimg)        
        y.append(steer)
        X.append(limg)
        # Augmentation to use left image - subtract an steering threshold  
        y.append(steer - angleTS)
        X.append(rimg)
        # Augmentation to use right image - sum an steering threshold
        y.append(steer + angleTS)
    #Shuffle the data before splitting
    X_train, y_train = shuffle(X, y)
    #Return data with 20% split for training
    return train_test_split(X_train, y_train,test_size=0.20)


files = ['driving_log_udacity.csv','driving_log.csv','driving_log_udacity.csv','driving_log.csv','driving_log_udacity.csv']
#Hyper parameters for each training
discard = [False,False,True,True,True]
batch_size = [32,32,32,32,32]
epochs = [3,3,3,3,6]
input_size=  (40, 160, 3);

model = vgg(input_size);
model.compile('adam', 'mse')
model.load_weights('model.h5')

for i in range(len(files)):
    X_train, X_valid, y_train, y_valid = loadcsv(files[i],discard[i])
    history = model.fit_generator(getData(X_train,y_train,input_size,batch_size[i]), nb_epoch=epochs[i],verbose=1,samples_per_epoch=len(X_train),validation_data = getData(X_valid,y_valid,input_size,batch_size[i]),nb_val_samples = len(X_valid))
    
with open('model.json', 'w') as f:
    f.write(model.to_json())
model.save_weights('model.h5')

    
