# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 14:05:38 2022

@author: fabia
"""

###############################################################################
# Library Imports

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten , Activation, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical 
import time
import pandas as pd
import os
import shutil
import datetime as dt


###############################################################################
# User Parameters

do_training = True # [CAN BE CHANGED]
do_visualization = True # [CAN BE CHANGED]

# if new classification type should be done
distribute_pictures_to_classification_folders = False
 

# data selection
start_year = 2005
train_years = 3
validation_years =3 # could be lower and train years increased
test_years = 3


# orignal picture size
original_width = 704 #pixel
original_height = 513 #pixel


# define constants
# for binary classification
basepath= "D:\\DeepLearning\\500hPaGeopotential\\binary_classification_fast\\"  # [TO BE ADPATED ON YOUR LOCAL MACHINE]
# for binary classification grayscale
#basepath = "D:\\DeepLearning\\500hPaGeopotential\\binary_classification_grayscale\\" # [TO BE ADPATED ON YOUR LOCAL MACHINE]


# for categorical classification [NOT YET IMPLEMENTED]
#basepath = "D:\\DeepLearning\\500hPaGeopotential\\categorical_classification\\"
# for categorical classification with forecast (D+1) [NOT YET IMPLEMENTED]
#basepath = "D:\\DeepLearning\\500hPaGeopotential\\categorical_classification_forecast\\""


# path to orignal map data
datapath = "D:\\DeepLearning\\500hPaGeopotential\\original_data\\" # [TO BE ADPATED ON YOUR LOCAL MACHINE]


# derived paths
trainpath = basepath + "train\\"
testpath = basepath + "test\\"
validationpath = basepath + "validation\\"

# label file name
csvfile = 'labels.csv'

# model parameters
n_epochs = 5 # numer of training epochs  [CAN BE CHANGED]
batch_size = 30 # [CAN BE CHANGED]
color_dim = 3 # for grayscale mode set to 1, for RGB color set to 3 [CAN BE CHANGED]
target_width = 400 # pixcel [do not change for now]
target_height = 250 #pixel [do not change for now]
classification_mode = 'binary' # categorical
n_labels = 2 # for binary classification

###############################################################################
# Functions

# function saves images to correct label subfolders and does cropping if 
# necessary
def move_images_to_folder(labels,startyear, endyear,type, color_dim, datapath, basepath):
    
    
    for y in range(startyear,endyear):
        fpath = datapath + str(y) + '\\'
        print(y)
        for file in os.listdir(fpath):
            
            #get label
            ts = pd.to_datetime(os.fsdecode(file).replace("geopotential_","").replace(".gif",""))
            
            im = Image.open(fpath + file)
            im = im.crop((100,100,500,350))
            # The crop rectangle, as a (left, upper, right, lower)-tuple. 
            
            # grayscale necessary?
            if color_dim == 1:
                im = im.convert('LA')
            
            
            # only maps that can be labeled, should be used
            if ts in labels.index:
            
                lab = labels.loc[ts]['WindLabel']
                #print("Label: " + str(lab))
                
                #determine destination
                destf=basepath + type + '\\' + str(lab) 
                isExist = os.path.exists(destf)
                if not isExist:
                    os.makedirs(destf)
                
                # get number of files in target folder
                nfiles = len(os.listdir(destf))
                          
                #move picture to corresponding folder
                #shutil.copyfile(fpath + file,destf + '\\' + str(nfiles) + '.png')
            
                # save picture
                im.save(destf + '\\' + str(nfiles) + '.png')


# determine number of training, test and validation data
def determine_number_of_data_points(sub_folder,base_folder, n_labels):
    
    count = 0
    for l in range(n_labels):
        count += len(os.listdir(base_folder + sub_folder + "\\" + str(l)))
    
    return count
    
    

###############################################################################
# Main Code    

if do_training:

    # read in label files
    labels = pd.read_csv(basepath + csvfile, sep = ',')
    labels.drop(columns = ['WindSpeed'], inplace=True)
    labels.set_index('TimeStamp', inplace=True)
    labels.index = pd.to_datetime(labels.index)
    
    
    # if necessary, sort pictures to corresponding sub folders with labels
    if distribute_pictures_to_classification_folders:
        
        move_images_to_folder(labels,start_year,start_year+train_years,'train',color_dim, datapath, basepath)
        move_images_to_folder(labels,start_year+train_years,start_year+train_years+validation_years, 'validation',color_dim, datapath, basepath)
        move_images_to_folder(labels,start_year+train_years+validation_years,start_year+train_years+validation_years+test_years, 'test',color_dim, datapath, basepath)
    
    
    
    # Training and validation flows
    train_datagen = ImageDataGenerator(rescale = 1./255,)
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    # determine number of data points
    n_points_training = determine_number_of_data_points("traiN", basepath, n_labels)
    n_points_validation = determine_number_of_data_points("validation", basepath, n_labels)
    n_points_test = determine_number_of_data_points("test", basepath, n_labels)
    
    
    # determine training and validation generator
    train_generator = train_datagen.flow_from_directory(       
        trainpath,
        color_mode = 'grayscale' if color_dim == 1 else 'rgb',
        batch_size=batch_size,
        target_size=(target_width,target_height),
        class_mode=classification_mode) #alternative categorical
    
    
    validation_generator = validation_datagen.flow_from_directory(
      validationpath,
        color_mode = 'grayscale' if color_dim == 1 else 'rgb',
        batch_size=batch_size,
        target_size=(target_width,target_height),
        class_mode=classification_mode)
    
    
    
    # Building the Deep Learning Model
    cnn = Sequential()
    cnn.add(Convolution2D(64,(3,3), input_shape = (target_width, target_height,color_dim))) 
    cnn.add(MaxPooling2D((2,2)))
    cnn.add(Convolution2D(64,(3,3),activation="relu"))
    cnn.add(MaxPooling2D((2,2)))   
    cnn.add(Convolution2D(64,(3,3),activation="relu")) 
    cnn.add(Flatten())
    cnn.add(Dense(64, activation='relu'))
    cnn.add(Dense(1 if classification_mode == 'binary' else n_labels, activation='sigmoid' if classification_mode == 'binary' else 'softmax')) #softmax for categorical
    cnn.compile(loss='binary_crossentropy' if classification_mode == 'binary' else 'categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    
    start = time.time()
    history = cnn.fit_generator(    
        train_generator,
        steps_per_epoch=n_points_training//batch_size,
        epochs=n_epochs,
        validation_data=validation_generator,
        validation_steps=n_points_validation//batch_size)
    end = time.time()
    print('Processing time [min]:',(end - start)/60)
    
    
    # save model
    cnn.save_weights(basepath +  dt.datetime.now().strftime('%Y%m%d%H%M') + '_cnn_weights.h5')
    cnn.save(basepath + dt.datetime.now().strftime('%Y%m%d%H%M')+ '_cnn_model')

# Visualizations
if do_visualization:
    
    if do_training==False:
        
        # load already trained model
        cnn = keras.models.load_model(basepath + '\\202203300236_cnn_model')
        cnn.load_weights(basepath + '202203300236_cnn_weights.h5')
        cnn.compile(loss= 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
       
            
    else:
        a =2
        # work with current model
        
    # Plot Accuracy [NOT TESTED YET]
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
    # Plot Loss [TO BE DONE]
    
    # Show Confusion Matrix [TO BE DONE]

    # Potential Additional Visualizations [TO BE DONE]
    