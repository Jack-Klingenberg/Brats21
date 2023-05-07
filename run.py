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
from src.utils import cli_out, logpath, output_hardware_check, generate_output_dir
from src.utils import dice_coef, precision, sensitivity, specificity, dice_coef_necrotic, dice_coef_edema ,dice_coef_enhancing
from src.utils import get_ids, get_dirnames, dir_path, file_path
from src.UNet import UNET

np.set_printoptions(precision=3, suppress=True)

parser = argparse.ArgumentParser(prog='BraTS2021 Project', description='Program to create, train, and test a U-net based architecture for the BraTS2021 image segmentation challenge', epilog='>=<=>=<=>=<')

# Set command line arguments 
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
parser.add_argument("--SMOOTH", action=argparse.BooleanOptionalAction, default=False)

args = vars(parser.parse_args())
#print(args)

# Set parameters 
IMG_SIZE=128
VOLUME_SLICES = 100 
VOLUME_START_AT = 22 
TRAIN_DATASET_PATH = args["PATH"]
SEGMENT_CLASSES = {0 : 'NOT tumor',   1 : 'NECROTIC/CORE', 2 : 'EDEMA',  3 : 'ENHANCING'}
METRIC_MAP = ['loss', 'accuracy','iou', 'dice_coef', "precision", "sensitivity", "specificity", "dice_coef_necrotic", "dice_coef_edema" ,"dice_coef_enhancing"]

# Define data generator class to feed data to the keras model during training
class DataGenerator(tf.keras.utils.Sequence):
    # Initialize Keras required functionality
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
        # Define outputs
        X = np.zeros((self.batch_size*VOLUME_SLICES, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size*VOLUME_SLICES, 240, 240))
        Y = np.zeros((self.batch_size*VOLUME_SLICES, *self.dim, 4))

        for index, id in enumerate(ids):
            example_path = os.path.join(TRAIN_DATASET_PATH, id)

            # Get data paths by id
            flair_image = nib.load(os.path.join(example_path, f'{id}_flair.nii.gz')).get_fdata()    
            ce_image = nib.load(os.path.join(example_path, f'{id}_t1ce.nii.gz')).get_fdata()
            seg_image = nib.load(os.path.join(example_path, f'{id}_seg.nii.gz')).get_fdata()

            # Resize images
            for j in range(VOLUME_SLICES):
                X[j +VOLUME_SLICES*index,:,:,0] = cv2.resize(flair_image[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
                X[j +VOLUME_SLICES*index,:,:,1] = cv2.resize(ce_image[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
                if(self.test):
                    y[j +VOLUME_SLICES*index] = seg_image[:,:,j+VOLUME_START_AT]

        # Use one hot encoding to encode labels
        if(self.test):
            y[y==4] = 3
            mask = tf.one_hot(y, 4)
            Y = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE))

        return X/np.max(X), Y

# If training is specified in command line
if args["TRAIN"]:
    # Check hardware to ensure that GPU devices are detected if applicable
    cli_out("Beginning hardware check.", padf=True)
    output_hardware_check()

    cli_out("Model compiling...")

    # Define model and compile it
    input_layer = Input((IMG_SIZE, IMG_SIZE, 2))
    model = UNET(input_layer)
    model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics = ['accuracy',tf.keras.metrics.MeanIoU(num_classes=4), dice_coef, precision, sensitivity, specificity, dice_coef_necrotic, dice_coef_edema ,dice_coef_enhancing] )


    # Get train, test, val split
    train_and_val_directories = [f.path for f in os.scandir(args["PATH"]) if f.is_dir()]
    train_and_test_ids = get_ids(train_and_val_directories)

    train_test_ids, val_ids = train_test_split(train_and_test_ids,test_size=0.2) 
    train_ids, test_ids = train_test_split(train_test_ids,test_size=0.15) 
    
    cli_out("Data split into train/test/val")

    # Feed ids to generator 
    training_generator = DataGenerator(train_ids)
    valid_generator = DataGenerator(val_ids)
    test_generator = DataGenerator(test_ids)

    # Define the path for the log
    log_path = logpath()
    os.mkdir(log_path)
    
    # Set up trainging sepcs
    filepath="UNet-improvement-{epoch:02d}-{val_accuracy:.3f}.hdf5" 
    cp = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
    logger = CSVLogger(log_path + 'unet.log')

    cli_out("About to begin training...")

    # Train!
    history =  model.fit(training_generator,
        epochs=args["EPOCHS"],
        callbacks= [stop, logger, cp],
        steps_per_epoch=len(train_ids),
        validation_data = valid_generator
    )  

    # Save the model
    save_path = args["SAVE"]
    cli_out("Model training complete, saving to: " + save_path)
    model.save(save_path)

    cli_out("Exiting...")
    sys.exit(0)

else:
    # If either testing or evaluation
    cli_out("Beginning hardware check." , padf=True)
    output_hardware_check()

    # Define where the model is and load it 
    model_path = args["MODEL"]

    model = keras.models.load_model(model_path, custom_objects={"dice_coef": dice_coef, "precision":precision, "sensitivity":sensitivity, "specificity":specificity, "dice_coef_necrotic":dice_coef_necrotic,"dice_coef_edema":dice_coef_edema, "dice_coef_enhancing":dice_coef_enhancing })

    # Find the evaluation/prediction data
    dirnames = get_dirnames(args["PATH"])
    data_generator = DataGenerator(dirnames, test = args["EVALUATE"])

    # If we want to predict...
    if(args["PREDICT"]):
        # Use model to predict 
        preds = model.predict(data_generator)

        # Make output directory for predictions
        dirname = generate_output_dir()
        if(os.path.isdir(os.path.join(args["OUTPUT"], dirname))):
            os.removedirs(os.path.join(args["OUTPUT"], dirname))
        os.makedirs(os.path.join(args["OUTPUT"], dirname))
        
        # Unsmoosh the different prediction masks and save them, perhaps forcing them to be binary output if the flag is selected
        for i in range(len(dirnames)):
            mask = preds[i*VOLUME_SLICES:(i+1)*VOLUME_SLICES,:,:,:]

            if(args["SMOOTH"]):
                for c in range(mask.shape[3]):
                    for j in range(mask.shape[0]):
                        layer = mask[j,:,:,c]
                        std = layer.flatten().std()
                        avg = layer.flatten().mean()
                        layer = (layer >= (avg+1*std))
                        mask[j,:,:,c] = layer

            np.save(os.path.join(args["OUTPUT"], dirname, dirnames[i]), mask)
    
        cli_out(f'Outputs saved to {os.path.join(args["OUTPUT"], dirname)}. Exiting...', padf=True)
        sys.exit(0)

    # If we want to evaluate...
    if(args["EVALUATE"]):
        # Evaluate the model
        metrics = model.evaluate(data_generator)

        # Output metrics
        cli_out("Keras Metrics",padf=True,padb=False)
        for i in range(len(METRIC_MAP)):
            cli_out(f'{METRIC_MAP[i]}:[{metrics[i]}]',padb=False)
        
        cli_out("Evaluation complete. Exiting...")
        sys.exit(0)