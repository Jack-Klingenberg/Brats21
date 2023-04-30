import tensorflow
import keras.backend as K

def iou(y_true, y_pred):
    y_true = tensorflow.keras.backend.flatten(y_true)
    y_pred = tensorflow.keras.backend.flatten(y_pred)
    y_true_f = tensorflow.cast(y_true, tensorflow.float32)
    y_pred_f = tensorflow.cast(y_pred, tensorflow.float32)
    intersection = tensorflow.keras.backend.sum(y_true_f * y_pred_f)
    union = tensorflow.keras.backend.sum(y_true_f) + tensorflow.keras.backend.sum(y_pred_f) - intersection
    return (intersection + 1e-7) / (union + 1e-7)

def iou_loss(y_true, y_pred):
    return -1*iou(y_true, y_pred)

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)