import os
import time
import argparse
from functools import partialmethod

parser = argparse.ArgumentParser(description='Train using either Alexnet Architecture or Nvidia Architecture')
parser.add_argument('--activation',type=str,default='elu',help='Give your activation function. Default is elu.')
parser.add_argument('--epochs',type=int,default=5,help='Number of epochs you want to train. Default is 1.')
parser.add_argument('--retrain',type=bool,default=False,help='Fresh train Or Retrain. Optional')
parser.add_argument('--batch_size',type=int,default=128,help='Specify your batch size,default 128.')
parser.add_argument('--datadir',type=str,default='.',help='Specify your data path')
parser.add_argument('--model_arch',type=str,default='nvidia',help="Either 'nvidia' architecture Or 'alexnet' arch")
parser.add_argument('--saved_model',type=str,default='.',help='If want to retrain then must give file path in hdf')
parser.add_argument('--validation_size',type=float,default=0.2,help='Size of validation set , default is 0.2')

args = parser.parse_args()

activation = args.activation
RETRAIN = args.retrain
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
DATA_DIR = args.datadir
model_arch = args.model_arch
saved_model_path = args.saved_model
test_size = args.validation_size


def argument_checker():

    if RETRAIN:
        if not os.path.exists(saved_model_path):
            raise FileNotFoundError('Specified Path not Found')

    if activation not in ['elu','relu','sigmoid','tanh','linear']:
        raise ValueError('Activation must be one of elu,relu,sigmoid,tanh,linear')

    if model_arch not in ['nvidia','alexnet']:
        raise ValueError('Model architecture must be one of "nvidia" or "alexnet"')

    if not os.path.exists(os.path.join(DATA_DIR,'IMG')) or not os.path.exists(os.path.join(DATA_DIR,'driving_log.csv')):
        raise FileNotFoundError('Data directory path must contain IMG directory and driving_log.csv file')


argument_checker()


import numpy as np
import pandas as pd

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import keras

if keras.__version__!= '2.0.6':
    print('This model is developed using keras version 2.0.6')
    print('Previous versions may not work properly and versions 1.x.x will not work.')

from keras import backend as K
from keras.optimizers import Adam
from keras.models import Sequential
from keras.models import Model
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Lambda
from keras.layers import Cropping2D
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard


from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

IMG_DIR = os.path.join(DATA_DIR,'IMG')
CSV_LOG_PATH = os.path.join(DATA_DIR,'driving_log.csv')

CF = 0.27
LR = 1e-3
DECAY = 0.99
samples = pd.read_csv(CSV_LOG_PATH)


def correction_factor():
    pass

steering_center = samples.steering
steering_left = steering_center + CF 
steering_right = steering_center - CF

steering_angle = np.concatenate((steering_center,steering_left,steering_right))



center_images = list(samples.center)
left_images = list(samples.left)
right_images = list(samples.right)

sample_images = []
sample_images += center_images
sample_images += left_images
sample_images += right_images


# In[ ]:

X_data = sample_images
y_data = steering_angle

X_train,X_validation,y_train,y_validation = train_test_split(X_data,y_data,test_size=test_size)

X_train,y_train = shuffle(X_train,y_train)


# In[ ]:

def flipped(data):
    return np.flip(data,axis=1)

train_data = {'X':X_train,'y':y_train}
validation_data = {'X':X_validation,'y':y_validation}


# In[ ]:

def generator(data,batch_size):
    
    num_samples = len(data['X'])
    xdata,ydata = data['X'],data['y']
    
    while True:
        for offset in range(0,num_samples,batch_size):
            xbatch_samples = xdata[offset:offset+batch_size]
            ybatch_samples = ydata[offset:offset+batch_size]
            
            images = []
            for batch_sample in xbatch_samples:
                img = mpimg.imread(os.path.join(IMG_DIR,batch_sample))
                images.append(flipped(img))
                
            
            X_batch = np.array(images)
            y_batch = ybatch_samples*(-1)
            yield shuffle(X_batch,y_batch)


# In[ ]:

train_generator = generator(train_data,batch_size=BATCH_SIZE) 
validation_generator = generator(validation_data,batch_size=BATCH_SIZE)


# In[ ]:
class ModelArch(object):
    def __init__(self,model_arch='nvidia'):
        self.model_arch = model_arch

    def nvidia_arch(self,activation):

        model = Sequential()

        model.add(Cropping2D(cropping=((65,45),(0,0)),
                             input_shape=(160,320,3)))
        model.add(Lambda(lambda x:K.tf.image.resize_images(x,(66,200),
                                                           method=K.tf.image.ResizeMethod.NEAREST_NEIGHBOR)))
        model.add(Lambda(lambda x:((x-K.mean(x))/K.std(x))))

        model.add(Conv2D(24,(5,5),strides=(2,2),activation=activation,padding='valid'))
        model.add(Conv2D(36,(5,5),strides=(2,2),activation=activation,padding='valid'))
        model.add(Conv2D(48,(5,5),strides=(2,2),activation=activation,padding='valid'))
        model.add(Conv2D(64,(3,3),strides=(1,1),activation=activation,padding='valid'))
        model.add(Conv2D(64,(3,3),strides=(1,1),activation=activation,padding='valid'))

        model.add(Flatten())
        model.add(Dense(10,activation=activation))
        model.add(Dense(50,activation=activation))
        model.add(Dense(10,activation=activation))
        model.add(Dense(1))

        return model

    def alexnet_arch(self,activation):

        alexnet_model = Sequential()

        alexnet_model.add(Cropping2D(cropping=((65, 40), (0, 0)),
                                     input_shape=(160, 320, 3)))
        alexnet_model.add(Lambda(lambda x: K.tf.image.resize_images(x, (224, 224),
                                                                    method=K.tf.image.ResizeMethod.NEAREST_NEIGHBOR)))
        alexnet_model.add(Lambda(lambda x: ((x - K.mean(x)) / K.std(x))))

        alexnet_model.add(Conv2D(96,(11,11),strides=(4,4),padding='same',activation=activation))
        alexnet_model.add(BatchNormalization())
        alexnet_model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        alexnet_model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation=activation))
        alexnet_model.add(BatchNormalization())
        alexnet_model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        alexnet_model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation=activation))
        alexnet_model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation=activation))
        alexnet_model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation=activation))
        alexnet_model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        alexnet_model.add(Flatten())

        alexnet_model.add(Dense(100, activation=activation))
        alexnet_model.add(Dense(50, activation=activation))
        alexnet_model.add(Dense(10, activation=activation))
        alexnet_model.add(Dense(1))

        return alexnet_model

adam = Adam(lr=LR,decay=DECAY)

if RETRAIN == True:
    model = load_model(saved_model_path,compile=True)
elif model_arch == 'nvidia':
    model = ModelArch('nvidia')
    model = model.nvidia_arch(activation)
    model.compile(loss='mse', optimizer=adam)
elif model_arch == 'alexnet':
    model = ModelArch('alexnet')
    model = model.alexnet_arch(activation)
    model.compile(loss='mse', optimizer=adam)


print(model.summary())

# In[ ]:
try:
    history_object = model.fit_generator(train_generator,
                                         steps_per_epoch=len(X_train)/BATCH_SIZE,
                                         callbacks=[ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
                                                    TensorBoard(log_dir='./logdir/',
                                                    histogram_freq=1,
                                                    batch_size=128,
                                                    write_grads=True)],
                                         validation_data=validation_generator,
                                         validation_steps=len(X_validation)/BATCH_SIZE,
                                         epochs=EPOCHS)
except K.tf.errors.ResourceExhaustedError:
    print('Your device is running out of memory . Try using a lower batch size.')


model.save('model'+'_'+model_arch+'_'+str(EPOCHS)+'.h5')

print(history_object.history.keys())



plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set','validation set'],loc='upper right')
plt.show()

