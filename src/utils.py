from tensorflow.python.client import device_lib
import keras.backend as K
from datetime import datetime
import os
import numpy as np

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
        cli_out("No GPUs detected. Note that training will be slow.")
    else:
        cli_out("Detected devices:", padb=False)
        for gpu in gpus:
            cli_out(gpu, padf=False,padb=False)
        print()
 
def logpath(prefix="TRAININGLOG"):
    now = datetime.now()
    path = (os.getcwd() + "/output/" + prefix + "-" + now.isoformat() + "/")
    return path
    


##################
#    Metrics     #
##################

def dice_coef(y_true, y_pred, epsilon=1.0):
    N = 4
    aggragate_loss = 0
    for i in range(N):
        yt_vector = K.flatten(y_true[:,:,:,i])
        yp_vector = K.flatten(y_pred[:,:,:,i])
        aggragate_loss += ((2.0 * K.sum(yt_vector * yp_vector) + epsilon) / (K.sum(yt_vector) + K.sum(yp_vector) + epsilon))
     
    avg = aggragate_loss / N
    return avg

def dice_coef_single_class(y_true,y_pred,i,epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,i] * y_pred[:,:,:,i]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,i])) + K.sum(K.square(y_pred[:,:,:,i])) + epsilon)

def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
    return dice_coef_single_class(y_true,y_pred, 1,epsilon=epsilon)

def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
    return dice_coef_single_class(y_true,y_pred, 2,epsilon=epsilon)

def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
    return dice_coef_single_class(y_true,y_pred, 3,epsilon=epsilon)

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

def getIDS(dirList):
    x = []
    for i in range(0,len(dirList)):
        x.append(dirList[i][dirList[i].rfind('/')+1:])
    return x