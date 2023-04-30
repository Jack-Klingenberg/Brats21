# Author: Jack Klingenberg
# Date: April 30th
# Description: Simple python script to add command line functionality

import matplotlib.pyplot as plt
import sys
sys.path.append('./src') 
import os 
from datetime import datetime
import keras
from Metrics import iou_loss
from Generators import DataGenerator3D

def usage():
    print("Usage: [--metrics?] [id]")

testing_ids = sys.argv

testing_ids.remove("run.py")
if(testing_ids[0] == "--metrics"):
    compute_metrics = True
    testing_ids.remove("--metrics")
else:
    comute_metrics = False

now = datetime.now()
os.makedirs("./output/" + now.strftime("OUTPUT%m-%d-%Y--%H:%M:%S/"))

network = keras.models.load_model('./models/vnet_dice.h5', custom_objects={"iou_loss": iou_loss})

image_size = 128
slices = 128 
start = 22
image_size=128
train_data_path = "./examples"
data_generator = DataGenerator3D(testing_ids,slices,start,image_size,train_data_path)

pred = network.predict(data_generator)

def getBinaryPrediction(masks):
    masks = masks[0,:,:,:,:]
    return -1*masks[:,:,25,0]




