#Importing Builtins
import os
import argparse


def parse_arguments():

    #Initilizing ArgumentParser() object
    parser = argparse.ArgumentParser(description='Train using either Alexnet Architecture or Nvidia Architecture')
    
    #Give the model architecture name , default is "alexnet"
    parser.add_argument('--model_arch',type=str,default='alexnet',help="Either 'nvidia' architecture Or 'alexnet' arch")
    
    #Want to retrain or not,default is false
    parser.add_argument('--retrain',type=bool,default=False,help='Fresh train Or Retrain. Optional')
    
    #If retrain then give the path of saved model path,default is the working directory
    parser.add_argument('--saved_model',type=str,default='.',help='If want to retrain then must give file path in hdf')
    
    #If not retrain then give the path of the data directory,default is the working directory
    parser.add_argument('--datadir',type=str,default='.',help='Specify your data path')
    
    #Specify the size of validation set,default is 0.2
    parser.add_argument('--validation_size',type=float,default=0.2,help='Size of validation set , default is 0.2')
    
    parser.add_argument('--batch_size',type=int,default=128,help='Specify your batch size,default 128.')
    
    #Specify the number  of epochs you want to train, default is 5
    parser.add_argument('--epochs',type=int,default=5,help='Number of epochs you want to train. Default is 1.')
    
    #Specify the activation function default is 'elu'
    parser.add_argument('--activation',type=str,default='elu',help='Give your activation function. Default is elu.')

    return parser 


arg_parser = parse_arguments()
args = parser.parse_args()

#Retreiving the parameters from the command line arguments
activation = args.activation
RETRAIN = args.retrain
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
DATA_DIR = args.datadir
model_arch = args.model_arch
saved_model_path = args.saved_model
test_size = args.validation_size

#Function for checking the arguments
def argument_checker():

    if model_arch not in ['nvidia','alexnet']:
        raise ValueError('Model architecture must be one of "nvidia" or "alexnet"')

    if RETRAIN:
        if not os.path.exists(saved_model_path):
            raise FileNotFoundError('Specified Path not Found')

    if not os.path.exists(os.path.join(DATA_DIR,'IMG')) or not os.path.exists(os.path.join(DATA_DIR,'driving_log.csv')):
        raise FileNotFoundError('Data directory path must contain IMG directory and driving_log.csv file')


    if activation not in ['elu','relu','sigmoid','tanh','linear']:
        raise ValueError('Activation must be one of elu,relu,sigmoid,tanh,linear')

#Cheeck the arguments passed
argument_checker()

#import libraries for number crunching and data exploration
import numpy as np
import pandas as pd

#import libraries for data visualization
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

#importing the deep learning library
import keras

#keras version checker
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

#Callback functions to be implemented later at training step
#Tensorboard visualization features added.
model_checkpt = ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5')
tensor_board = TensorBoard(log_dir='./logdir/',histogram_freq=1,batch_size=32,write_grads=True)


#Correction Factor for left and right images
CF = 0.27
#Learning rate 
LR = 1e-3
#Learning rate decay
DECAY = 0.99
#Reading the csv file
samples = pd.read_csv(CSV_LOG_PATH)

samples = pd.read_csv(CSV_LOG_PATH,header=None)
columns = ['center','left','right','steering','throttle','brake','speed']
samples.columns = columns

# as the recorded image names in driving log file contains the whole directory path
#function for removing the directory path
def path_remover(path):
    return path.split('/')[-1]

samples.center = list(map(path_remover,samples.center))
samples.left = list(map(path_remover,samples.left))
samples.right = list(map(path_remover,samples.right))

print(samples.head())

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

#adding the flipped images with prefix flipped_ to the original image names
#at time of training generator function can understand which image is to flip
flipped_images = list(map(lambda x:'flipped_'+ x,sample_images))
sample_images = sample_images + flipped_images
steering_angle = np.concatenate((steering_angle,steering_angle*(-1)))

X_data = sample_images
y_data = steering_angle

#splitting the training data into training and validation split
X_train,X_validation,y_train,y_validation = train_test_split(X_data,y_data,test_size=test_size)

#shuffling the training data
X_train,y_train = shuffle(X_train,y_train)

#function for flipping the image
def flipped(data):
    return np.flip(data,axis=1)

#generator function for feeding the training operation for a fixed batch size
def generator(data,batch_size):
    
    num_samples = len(data['X'])
    xdata,ydata = data['X'],data['y']
    
    while True:
        for offset in range(0,num_samples,batch_size):
            xbatch_samples = xdata[offset:offset+batch_size]
            ybatch_samples = ydata[offset:offset+batch_size]
            
            images = []
            for batch_sample in xbatch_samples:
                if batch_sample.startswith('flipped_'):
                    img = mpimg.imread(os.path.join(IMG_DIR,batch_sample.replace('flipped_','')))
                    img = flipped(img)
                    images.append(img)
                else:
                    img = mpimg.imread(os.path.join(IMG_DIR,batch_sample))
                    images.append(img)
                
            
            X_batch = np.array(images)
            y_batch = ybatch_samples*(-1)
            yield shuffle(X_batch,y_batch)


train_data = {'X':X_train,'y':y_train}
validation_data = {'X':X_validation,'y':y_validation}

train_generator = generator(train_data,batch_size=BATCH_SIZE) 
validation_generator = generator(validation_data,batch_size=BATCH_SIZE)

#The ModelArch class which contains the both model methods nvidia_arch() and alexnet_arch()
class ModelArch(object):
    """This model class contains two architecture:
       1. Nvidia Architecture:
       2. Alexnet Architecture:
    """
    def __init__(self,model_arch='nvidia'):
        self.model_arch = model_arch

    def nvidia_arch(self,activation):
        """This model contains these layers as folows :
           Preprocessing layers :
           ---------------------
           (1) First a cropping layer which cropped out the unnecessary background from the input image.
           (2) This layer is lambda layer which resized the images to 66x200. Which is the required input
               shape for this architecture.For resizing the nearest neighbor approach is used.
           (3) Third layer is the normalizing layer which is also a lambda layer.

           Convolution layers:
           -------------------
           There are total 5 convolution layers and all of which have valid padding.This model is trained
           using activation function 'elu'.
           first_layer : 24 filters with size of 5x5 with strides 2x2
           second_layer: 36 filters with size of 5x5 with strides 2x2
           third_layer: 48 filters with size of 5x5 with strides 2x2
           fourth_layer: 64 filters with size of 5x5 with strides 1x1
           fifth_layer: 64 filters with size of 5x5 with strides 1x1

           Fully Connected layers:
           -----------------------
           A total of 4 fully connected layers with output sizes 100,50,10,1 respectively applied.
           The last layer is the output layer.
           So no activation is applied.
        """
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
        model.add(Dense(100,activation=activation))
        model.add(Dense(50,activation=activation))
        model.add(Dense(10,activation=activation))
        model.add(Dense(1))

        return model

    def alexnet_arch(self,activation):
        """
           Orignial Alexnet Architecture popularize the use of 'relu' but here 'elu' is used.
           I have also ommitted the last three fully connected layers to fit  the output .
           This model contains these layers as folows :
           Preprocessing layers :
           ---------------------
           (1) First a cropping layer which cropped out the unnecessary background from the input image.
           (2) This layer is lambda layer which resized the images to 224x224. Which is the required input
               shape for this architecture.For resizing the nearest neighbor approach is used.
           (3) Third layer is the normalizing layer which is also a lambda layer.

           Convolution layers:
           -------------------
           Here all paddings are same.
           In first conv layer 96 11x11 filters  with strides 4x4 is applied followed by a Batch normalization layer
           and a MaxPooling layer with pool size 3x3 and strides 2x2 .
           In second conv layer 256 5x5 filters  with strides 1x1 is applied followed by a Batch normalization layer
           and a MaxPooling layer with pool size 3x3 and strides 2x2 .
           In third, fourth, and fifth conv layer 384,384,256 3x3 filters with strides 1x1 is applied respectively,
           followed by a MaxPooling layer with pool size 3x3 and strides 2x2 .
         
           Fully Connected layers:
           -----------------------
           A total of 4 fully connected layers with output sizes 100,50,10,1 respectively applied.
           The last layer is the output layer.
           So no activation is applied.
        """
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

#Initializing the adam optimizer with learning rate and decay parameter
adam = Adam(lr=LR,decay=DECAY)

#If want to retrain then there is no need to create an instance of ModelArch() class.
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

#If for larger batch size the device is running out of mememory
#Then it stops the program.
try:
    #creating a history object for visualization later 
    history_object = model.fit_generator(train_generator,
                                         steps_per_epoch=len(X_train)/BATCH_SIZE,
                                         callbacks=[model_checkpt,tensor_board],
                                         validation_data=validation_generator,
                                         validation_steps=len(X_validation)/BATCH_SIZE,
                                         epochs=EPOCHS)
if K.tf.errors.ResourceExhaustedError:
    print('Your device is running out of memory . Try using a lower batch size.')
    break
#For tensor board visualization go to terminal and type $ tensorboard --logdir=$[PATH_TO_YOUR_LOGDIR]
#Saving the model
model.save('model'+'_'+model_arch+'_'+str(EPOCHS)+'.h5')

print(history_object.history.keys())

#Loss visualization
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set','validation set'],loc='upper right')
plt.show()

