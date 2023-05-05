# Misc Packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import argparse
import sys
import numpy as np

# Imaging Packages
import nibabel as nib
import cv2

# Machine Learning Packages
import keras
from keras.callbacks import CSVLogger
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Local functions
from src.utils import cli_out, convex_hull, logpath, output_hardware_check, generate_output_dir
from src.utils import dice_coef, precision, sensitivity, specificity, dice_coef_necrotic, dice_coef_edema ,dice_coef_enhancing
from src.utils import get_ids, get_dirnames, dir_path, file_path
from src.UNet import UNET

np.set_printoptions(precision=3, suppress=True)

parser = argparse.ArgumentParser(prog='BraTS2021 Project', description='Program to create, train, and test a U-net based architecture for the BraTS2021 image segmentation challenge', epilog='>=<=>=<=>=<')

group = parser.add_mutually_exclusive_group()
group.add_argument("-P", "--PREDICT", action=argparse.BooleanOptionalAction)
group.add_argument("-E", "--EVALUATE", action=argparse.BooleanOptionalAction)
group.add_argument("-T", "--TRAIN", action=argparse.BooleanOptionalAction)

parser.add_argument("--PATH", default='./data/',type=dir_path)
parser.add_argument("--MODEL", default='./models/UnetV4.h5',type=file_path)
parser.add_argument("--LOG", default="./output/", type=dir_path)
parser.add_argument("--OUTPUT", default="./output/", type=dir_path)
parser.add_argument("--EPOCHS", default=10, type=int)
parser.add_argument("--SAVE", default="./models/unet.h5")
parser.add_argument("--CONVEX", action=argparse.BooleanOptionalAction, default=False)

args = vars(parser.parse_args())
#print(args)

IMG_SIZE=128
VOLUME_SLICES = 100 
VOLUME_START_AT = 22 
TRAIN_DATASET_PATH = args["PATH"]
SEGMENT_CLASSES = {0 : 'NOT tumor',   1 : 'NECROTIC/CORE', 2 : 'EDEMA',  3 : 'ENHANCING'}
METRIC_MAP = ['loss', 'accuracy','iou', 'dice_coef', "precision", "sensitivity", "specificity", "dice_coef_necrotic", "dice_coef_edema" ,"dice_coef_enhancing"]

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, dim=(IMG_SIZE,IMG_SIZE), batch_size = 1, n_channels = 2, shuffle=True, test=False):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.test = test
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
                X[j +VOLUME_SLICES*index,:,:,0] = cv2.resize(flair_image[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
                X[j +VOLUME_SLICES*index,:,:,1] = cv2.resize(ce_image[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
                if(self.test):
                    y[j +VOLUME_SLICES*index] = seg_image[:,:,j+VOLUME_START_AT]

        if(self.test):
            y[y==4] = 3
            mask = tf.one_hot(y, 4)
            Y = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE))

        return X/np.max(X), Y

if args["TRAIN"]:
    cli_out("Beginning hardware check.", padf=True)
    output_hardware_check()

    if(args["MODEL"] == None or not os.path.isfile(args["MODEL"])):
        if(args["MODEL"] != None):
            cli_out("Model not recognized. Training with u-net architecture...")

        cli_out("Model compiled")

        input_layer = Input((IMG_SIZE, IMG_SIZE, 2))
        model = UNET(input_layer, 'he_normal', 0.2)
        model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics = ['accuracy',tf.keras.metrics.MeanIoU(num_classes=4), dice_coef, precision, sensitivity, specificity, dice_coef_necrotic, dice_coef_edema ,dice_coef_enhancing] )

        train_and_val_directories = [f.path for f in os.scandir(args["PATH"]) if f.is_dir()]
        train_and_test_ids = get_ids(train_and_val_directories)

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

        save_path = args["SAVE"]
        cli_out("Model training complete, saving to: " + save_path)
        model.save(save_path)

        cli_out("Exiting...")
        sys.exit(0)

else:
    cli_out("Beginning hardware check." , padf=True)
    output_hardware_check()

    model_path = args["MODEL"]

    model = keras.models.load_model(model_path, custom_objects={"dice_coef": dice_coef, "precision":precision, "sensitivity":sensitivity, "specificity":specificity, "dice_coef_necrotic":dice_coef_necrotic,"dice_coef_edema":dice_coef_edema, "dice_coef_enhancing":dice_coef_enhancing })

    dirnames = get_dirnames(args["PATH"])
    data_generator = DataGenerator(dirnames, test = args["EVALUATE"])

    if(args["PREDICT"]):
        preds = model.predict(data_generator)

        dirname = generate_output_dir()
        if(os.path.isdir(os.path.join(args["OUTPUT"], dirname))):
            os.removedirs(os.path.join(args["OUTPUT"], dirname))
        os.makedirs(os.path.join(args["OUTPUT"], dirname))

        for i in range(len(dirnames)):
            mask = preds[i*VOLUME_SLICES:(i+1)*VOLUME_SLICES,:,:,]

            if(args["--CONVEX"]):
                mask = convex_hull(mask)

            np.save(os.path.join(args["OUTPUT"], dirname, dirnames[i]), )
    

        cli_out("Outputs saved. Exiting...", padf=True)
        sys.exit(0)

    if(args["EVALUATE"]):
        metrics = model.evaluate(data_generator)

        cli_out("Keras Metrics",padf=True,padb=False)
        for i in range(len(METRIC_MAP)):
            cli_out(f'{METRIC_MAP[i]}:[{metrics[i]}]',padb=False)
        
        cli_out("Single class metrics")

        cli_out("Evaluation complete. Exiting...")
        sys.exit(0)