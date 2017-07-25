# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

### This project consists of the following steps:

*  Data Collection by running the simulator.
*  Data Exploration For better understanding.
*  Building models that can predict steering angles.
*  Testing the model and recording the video.
*  Discussion about the shortcomings of this approach.

**File Specification :**

These project repository consists of 6 main files .

           1.  model.py : Python file for running the training operation either from start or load a
                          pretrained model to carry out further training.
                           
           2.  behavorial_cloning.ipynb : A ipython notebook has been included to better understand the
                                          model parameters,data exploration and training time , loading
                                          a pretrained model etc.(It is not required by udacity.)
           
           3.  drive.py : Python file for feeding the simulator with prediction data .
           
           4.  video.py : Python file for making video from recorded image through testing the model.
           5.  model.h5 : The trained model was saved in hdf file format by keras . This file is
                          required for sucessfully running the simulator in autonomous mode.
                          
           6.  video_track1.mp4 : The video made by video.py using the data captured by the drive.py 
                                  during the driving in autonomous mode for track one.
                                  
**Hardware an environment specification :**
 
 * OS used : Ubuntu Linux 16.04 LTS
 * This project is done in python version 3.6.1 .
 * Tensorflow version 1.2.1 is used. (Gpu version and compiled from source)
 * keras version 2.0.6 is used. (keras version 1.x.x will not work)
 * GEFORCE GTX 1050 gpu with memory 4 GB is used for training . (Cuda compute capability of 6.1)

**Usage**

* Suppose the data directory named 'data' is in your working directory and contains the 'IMG' directory 
  and 'driving_log.csv' file Want to train it with alexnet architecture. Enter the command in a terminal.
         python model.py --model_arch='alexnet' --retrain=False --data_dir=data --batch_size=32

* After successfull training, let the model has been saved in working directory named 'model.h5' and 
  to drive the car in autonomous mode and caputure the run in a 'run1' directory,try it in a terminal
         python drive.py model.h5 run1

* For making video from the directory 'run1',try it in a terminal
         python video.py run1 --fps=48
 
 **Data Collection :**
  
  * The obvious approach for training deep neural network to work better is to train it on more data.
  * Around 153k images have been used for training .
  * Images are from both two tracks with 70:30(track1:track2) proportion.
  * All center,left,right images are used .
  * And images were flipped for data augmentation .
  
  **Data Exploration and some preprocessing :**

  
  **Model Specification :**
  Here I mainly tried 2 different models for training.
  
  **NVIDIA MODEL ARCHITECTURE :**
  
  First is NVIDIA convnet architecture which was used by Nvidia to train and test on real autonomous car .
  ![image:](./resources/cnn-architecture-624x890.png)
  
  **ALEXNET ARCHITECTURE :**
  ![image:](./resources/cnn-architecture-624x890.png)


