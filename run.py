import os
import cv2
import argparse
import glob
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import data
from skimage.util import montage 
import skimage.transform as skTrans
from skimage.transform import rotate
from skimage.transform import resize

# Imaging Packages
import nilearn as nl
import nibabel as nib
import nilearn.plotting as nlplt


# Machine Learning Packages
import keras
import keras.backend as K
from keras.callbacks import CSVLogger
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.layers.experimental import preprocessing
from keras.callbacks import ModelCheckpoint, EarlyStopping


# Local functions
from src.utils import cli_out, logpath, output_hardware_check
from src.utils import dice_coef, precision, sensitivity, specificity, dice_coef_necrotic, dice_coef_edema ,dice_coef_enhancing
from src.utils import getIDS
from src.UNet import UNET

# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

parser = argparse.ArgumentParser(prog='BraTS2021 Project', description='Program to create, train, and test a U-net based architecture for the BraTS2021 image segmentation challenge', epilog='>=<=>=<=>=<')
parser.add_argument("-P", "--PREDICT", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--PATH", default='./data/')
parser.add_argument("--MODEL")
parser.add_argument("--LOG", default="./output/")
parser.add_argument("--EPOCHS", default=10, type=int)

args = vars(parser.parse_args())

IMG_SIZE=128
TRAIN_DATASET_PATH = args["PATH"]
VOLUME_SLICES = 100 
VOLUME_START_AT = 22 

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, dim=(IMG_SIZE,IMG_SIZE), batch_size = 1, n_channels = 2, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        Batch_ids = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(Batch_ids)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, ids):
        X = np.zeros((self.batch_size*VOLUME_SLICES, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size*VOLUME_SLICES, 240, 240))
        Y = np.zeros((self.batch_size*VOLUME_SLICES, *self.dim, 4))

        for index, id in enumerate(ids):
            example_path = os.path.join(TRAIN_DATASET_PATH, id)

            flair_image = nib.load(os.path.join(example_path, f'{id}_flair.nii.gz')).get_fdata()    
            ce_image = nib.load(os.path.join(example_path, f'{id}_t1ce.nii.gz')).get_fdata()
            seg_image = nib.load(os.path.join(example_path, f'{id}_seg.nii.gz')).get_fdata()

            for j in range(VOLUME_SLICES):
                X[j +VOLUME_SLICES*index,:,:,0] = cv2.resize(flair_image[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE));
                X[j +VOLUME_SLICES*index,:,:,1] = cv2.resize(ce_image[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
                y[j +VOLUME_SLICES*index] = seg_image[:,:,j+VOLUME_START_AT]
            
        y[y==4] = 3;
        mask = tf.one_hot(y, 4);
        Y = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE));
        return X/np.max(X), Y
        
if not args["PREDICT"]:
    cli_out("Beginning training run. Beginning hardware check.", padf=True)
    output_hardware_check()

    if(args["MODEL"] == None or not os.path.isfile(args["MODEL"])):
        if(args["MODEL"] != None):
            cli_out("Model not recognized. Training with u-net architecture...")

        cli_out("Model compiled")

        input_layer = Input((IMG_SIZE, IMG_SIZE, 2))
        model = UNET(input_layer, 'he_normal', 0.2)
        model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics = ['accuracy',tf.keras.metrics.MeanIoU(num_classes=4), dice_coef, precision, sensitivity, specificity, dice_coef_necrotic, dice_coef_edema ,dice_coef_enhancing] )

        train_and_val_directories = [f.path for f in os.scandir(args["PATH"]) if f.is_dir()]
        train_and_test_ids = getIDS(train_and_val_directories)

        train_test_ids, val_ids = train_test_split(train_and_test_ids,test_size=0.2) 
        train_ids, test_ids = train_test_split(train_test_ids,test_size=0.15) 
        
        cli_out("Data split into train/test/val")

        training_generator = DataGenerator(train_ids)
        valid_generator = DataGenerator(val_ids)
        test_generator = DataGenerator(test_ids)

        log_path = logpath()
        os.mkdir(log_path)
        
        filepath="3D-UNet-2018-weights-improvement-{epoch:02d}-{val_accuracy:.3f}.hdf5" 
        cp = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
        logger = CSVLogger(log_path + 'unet_training_log.log')

        cli_out("About to begin training...")

        history =  model.fit(training_generator,
            epochs=args["EPOCHS"],
            callbacks= [stop, logger, cp],
            steps_per_epoch=len(train_ids),
            validation_data = valid_generator
        )  

else:
    cli_out("Beginning testing run. Beginning hardware check." , padf=True)
    output_hardware_check()


