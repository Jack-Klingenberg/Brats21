from tensorflow.python.client import device_lib
import keras.backend as K
from datetime import datetime
import os
import numpy as np
import scipy
import tensorflow as tf

IMG_SIZE=128
VOLUME_SLICES = 100 
VOLUME_START_AT = 22 
BATCH_SIZE = 1
TRAIN_DATASET_PATH = "../data/"
SEGMENT_CLASSES = {0 : 'NOT tumor',   1 : 'NECROTIC/CORE', 2 : 'EDEMA',  3 : 'ENHANCING'}

##################
#      CLI       #
##################

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def cli_out(msg, padf=False, padb=True):
    if(padf and padb):
        print("\n  >>> " + msg + "\n")
        return
    if(padf and not padb):
        print("\n  >>> " + msg)
        return
    if(padb and not padf):
        print("  >>> " + msg + "\n")
        return
    print("  >>> " + msg)
    return

def output_hardware_check(): 
    gpus = get_available_gpus()

    if len(gpus) == 0:
        cli_out("No GPUs detected. Note that training/testing/evaluation may be slow.")
    else:
        cli_out("Detected devices:", padb=False)
        for gpu in gpus:
            cli_out(gpu, padf=False,padb=False)
        print()

# Define path for the log outputs
def logpath(prefix="TRAININGLOG"):
    now = datetime.now()
    path = (os.getcwd() + "/output/" + prefix + "-" + now.isoformat() + "/")
    return path
    


##################
#    Metrics     #
##################

# Compute the dice coefficient for a 4D Volume 
def dice_coef(y_true, y_pred, epsilon=1.0):
    N = 4
    aggragate_loss = 0
    for i in range(N):
        yt_vector = K.flatten(y_true[:,:,:,i])
        yp_vector = K.flatten(y_pred[:,:,:,i])
        aggragate_loss += ((2.0 * K.sum(yt_vector * yp_vector) + epsilon) / (K.sum(yt_vector) + K.sum(yp_vector) + epsilon))
     
    avg = aggragate_loss / N
    return avg

# Compute the general dice coeff for a certain dimension
def dice_coef_single_class(y_true,y_pred,i,epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,i] * y_pred[:,:,:,i]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,i])) + K.sum(K.square(y_pred[:,:,:,i])) + epsilon)

# Compute specific dice coeffs for different subregions
def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
    return dice_coef_single_class(y_true,y_pred, 1,epsilon=epsilon)

def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
    return dice_coef_single_class(y_true,y_pred, 2,epsilon=epsilon)

def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
    return dice_coef_single_class(y_true,y_pred, 3,epsilon=epsilon)

# Compute confusion matrix stats
def precision(y_true, y_pred):
    tpfn = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return TP(y_true,y_pred) / (tpfn + K.epsilon())
    
def sensitivity(y_true, y_pred):
    tpfn = K.sum(K.round(K.clip(y_true, 0, 1)))
    return TP(y_true, y_pred) / (tpfn + K.epsilon())

def specificity(y_true, y_pred):
    tnfp = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return TN(y_true,y_pred) / (tnfp + K.epsilon())

def TP(y_true,y_pred):
    return K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

def TN(y_true,y_pred):
    return K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))


# MISC 

def get_ids(dirList):
    x = []
    for i in range(0,len(dirList)):
        x.append(dirList[i][dirList[i].rfind('/')+1:])
    return x

def get_dirnames(root):
    paths = []
    for name in os.scandir(root):
        paths.append(name.name)
    return paths


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)
    
def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise NotADirectoryError(string)

'''def mask_to_convex_hull(mask):
    
    for i in range(mask.shape[0]):
'''

def generate_output_dir():
    now = datetime.now()
    return "PRED-" + str(now.isoformat())

'https://stackoverflow.com/questions/46310603/how-to-compute-convex-hull-image-volume-in-3d-numpy-arrays'
def flood_fill_hull(image):    
    points = np.transpose(np.where(image))
    hull = scipy.spatial.ConvexHull(points)
    deln = scipy.spatial.Delaunay(points[hull.vertices]) 
    idx = np.stack(np.indices(image.shape), axis = -1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.zeros(image.shape)
    out_img[out_idx] = 1
    return out_img, hull

def convex_hull(vol):
    hull_mask = np.zeros(vol.shape)
    for i in range(vol.shape[0]):
        if( np.count_nonzero(vol[i,:,:]) <= 5):
            hull_mask[i,:,:] = vol[i,:,:]
            continue
        arr,_ = flood_fill_hull(vol[i,:,:])
        hull_mask[i,:,:] = arr
    return hull_mask


