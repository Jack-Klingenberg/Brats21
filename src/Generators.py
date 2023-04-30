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
  def __init__(self, list_IDs, slices,start,image_size,train_data_path, batch_size = 1, n_channels = 2, shuffle=True, verbose=False):
    self.dim = (image_size,image_size)
    self.batch_size = batch_size
    self.list_IDs = list_IDs
    self.n_channels = n_channels
    self.shuffle = shuffle
    self.on_epoch_end()
    self.verbose = verbose
    self.log_dir = None

    self.slices = slices
    self.start = start
    self.image_size = image_size
    self.train_data_path = train_data_path


    if(self.verbose): 
      now = datetime.now()
      self.log_dir = "../output/" + now.strftime("GENERATOR_LOG%m-%d-%Y--%H:%M:%S/")

  
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
      
      layer = int(flair_image.shape[2]/2)
      if self.verbose and index==0:
        os.makedirs(self.log_dir)
        plt.imshow(flair_image[:,:,layer])
        plt.savefig(os.path.join(self.log_dir, "flair1.png"))
        plt.imshow(ce_image[:,:,layer])
        plt.savefig(os.path.join(self.log_dir, "ce1.png"))
        plt.imshow(seg_image[:,:,layer])
        plt.savefig(os.path.join(self.log_dir, "seg1.png"))

      # For each image, resize such that we omit useless channels (ie the very bottom of the brain scans that will only contain the table so no actual brain data.)
      for j in range(self.slices):
        X[j+(self.slices*index),:,:,0] = cv2.resize(flair_image[:,:,j+self.start], (self.image_size, self.image_size))
        X[j+(self.slices*index),:,:,1] = cv2.resize(ce_image[:,:,j+self.start], (self.image_size, self.image_size))
        y[j+(self.slices*index),:,:] = cv2.resize(seg_image[:,:,j+self.start], (self.image_size, self.image_size))


      # Reshape the X with an extra dimension (will be reduced later). Images are 128*128*128 blocks with 2 "channels" for the 
      # flair imaging and ce imaging (2 different types of MRI scans)
      X = X.reshape(1,128,128,128,2)
      y = y.reshape(1,128,128,128)

      # Use one hot encoding to get categorical labels for the different tumor types 
      y[y==4] = 3;
      y = tensorflow.one_hot(y, 4);
    
      if(self.verbose and index==0):
        plt.imshow(X[0,:,:,layer-self.start,0])
        plt.savefig(os.path.join(self.log_dir, "flair2.png"))
        plt.imshow(X[0,:,:,layer-self.start,1])
        plt.savefig(0,os.path.join(self.log_dir, "ce2.png"))
        plt.imshow(y[0,:,:,layer-self.start])
        plt.savefig(os.path.join(self.log_dir, "seg2.png"))


      # Normalize data, and return as tuple with labels
      return X/np.max(X), y
    