# Author: Jack Klingenberg
# Date: April 16th 
# Description: modularizing the data generator class so that 
# we do not have to definine it within the notebook.  

import keras
import os 
import numpy as np
import nibabel as nib
import tensorflow
import cv2
import matplotlib.pyplot as plt
from datetime import datetime

Sequence = keras.utils.Sequence

class DataGenerator3D(Sequence):
  # From stanford.edu
  def __init__(self, list_IDs, slices,start,image_size,train_data_path, batch_size = 1, n_channels = 2, shuffle=True):
    self.dim = (image_size,image_size)
    self.batch_size = batch_size
    self.list_IDs = list_IDs
    self.n_channels = n_channels
    self.shuffle = shuffle
    self.on_epoch_end()

    self.slices = slices
    self.start = start
    self.image_size = image_size
    self.train_data_path = train_data_path
  
  # From stanford.edu
  def __len__(self):
    return int(np.floor(len(self.list_IDs) / self.batch_size))

  # From stanford.edu
  def __getitem__(self, index):
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
    Batch_ids = [self.list_IDs[k] for k in indexes]
    X, y = self.__data_generation(Batch_ids)

    return X, y

  # From stanford.edu
  def on_epoch_end(self):
      'Updates indexes after each epoch'
      self.indexes = np.arange(len(self.list_IDs))
      if self.shuffle == True:
          np.random.shuffle(self.indexes)

  def __data_generation(self, ids):
    print(self.list_IDs)
    X = np.zeros((self.batch_size*self.slices, *self.dim, self.n_channels))
    y = np.zeros((self.batch_size*self.slices, 128, 128))

    # Generate data
    for index, id in enumerate(ids):
      
      example_path = os.path.join(self.train_data_path, id)
      

      flair_image = nib.load(os.path.join(example_path, f'{id}_flair.nii.gz')).get_fdata()    
      ce_image = nib.load(os.path.join(example_path, f'{id}_t1ce.nii.gz')).get_fdata()
      seg_image = nib.load(os.path.join(example_path, f'{id}_seg.nii.gz')).get_fdata()
    

      # For each image, resize such that we omit useless channels (ie the very bottom of the brain scans that will only contain the table so no actual brain data.)
      X[:,:,:,0] = flair_image[:,:,]
      X[:,:,:,1] = ce_image[:,:,]
      y[:,:,:] = seg_image[:,:,]


      # Reshape the X with an extra dimension (will be reduced later). Images are 128*128*128 blocks with 2 "channels" for the 
      # flair imaging and ce imaging (2 different types of MRI scans)
      X = X.reshape(1,128,128,128,2)
      y = y.reshape(1,128,128,128)

      # Use one hot encoding to get categorical labels for the different tumor types 
      y[y==4] = 3;
      y = tensorflow.one_hot(y, 4);

      # Normalize data, and return as tuple with labels
      return X/np.max(X), y
    