#################### IMPORTS #################### 
import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

import utils
import ni_utils

# from focal_loss.losses import *
# import dill

from deep_learning.DataGeneratorClass import *
from deep_learning.base_networks import *
# from deep_learning.parameters import *
from deep_learning.inception_networks import *
from deep_learning.residualAttentionNetworkModels import *
from deep_learning.loss_functions import quantile_loss
from keras.constraints import NonNeg

from dunet_model import *

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.losses import categorical_hinge, hinge, squared_hinge, hinge, binary_crossentropy
from CLR.clr_callback import *

#Set up GPU environment
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "4"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1


#################### LOSSES ####################
#Losses taken from https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))

    return 1 - numerator / denominator

def loss(y_true, y_pred):
    def dice_loss(y_true, y_pred):
        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
        denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))

        return tf.reshape(1 - numerator / denominator, (-1, 1, 1))

    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

#################### PARAMETERS AND DIRECTORIES ####################
#Parameters
train_params_multiclass = {'normalize': False,
          'batch_size': 16,
          'n_classes': 1,
          'n_channels': 1,
          'shuffle': True}

#Directories
BASE_DIR = '/nfsshare/gianca-group/dpena3/active_learning/ATLAS_feb2020'
DATA_DIR = ''
PERCENT_FOLDER = 'test5'
dataDir = BASE_DIR + '/pre-processing/data_model-ready/' + DATA_DIR
imgDir = dataDir

#Dataframe
fileIn = '../pre-processing/ATLAS_stroke_labels_20200122.csv'
# fileIn = '../pre-processing/ATLAS_stroke_regression_20200223.csv' #has lesion areas in it

#################### DATAFRAMES ####################
xlsFilepath = fileIn
patFr =  ni_utils.loadSubjGt(fileIn, 'stroke_ct') 
patFr = patFr[patFr['labels'] == 1][:600] #get only stroke
# patFr = patFr[patFr[patFr['label'] == 1]['lesion_area'].values > 1500] #get only large strokes
patIDList = patFr['filename'].tolist()
patFr_labels = patFr['labels'].tolist()

print('Number of subjects', len(patIDList))

#################### MODEL INPUTS ####################
#Imput image
# input_dim = (192, 192, 1) #regular unet
input_dim = (192, 192, 4) #dunet

input_dim_for_data_gen = input_dim

#Generator
train_generator = DataGenerator_stroke_d_unet(patIDList,
                            '',
                            data_dir=dataDir,
                            xls_filepath = xlsFilepath,
                            dim=input_dim_for_data_gen,
                            **train_params_multiclass)
#Regular unet DataGenerator_stroke_unet
#DUNET DataGenerator_stroke_d_unet

#Model
model = Unet3d()
model.compile(optimizer = Adam(lr = 1e-5), loss = dice_loss, metrics = ['accuracy'])
print(model.summary())
nEpochs = 30

#D_Unet
#Unet_origin
#Unet3d

#Model saving directory
network_model_dir = BASE_DIR + '/model/saved_models/'
FILEPATH_MODEL = "multiclass_weights_siamese_merge_L1_inception" + ".hdf5"
FILEPATH_MODEL = os.path.join(network_model_dir, PERCENT_FOLDER, FILEPATH_MODEL)

final_folder_path = os.path.join(network_model_dir, PERCENT_FOLDER)
if not os.path.exists(final_folder_path):
    os.makedirs(final_folder_path)

#Callbacks
callbacks_list = [ModelCheckpoint(FILEPATH_MODEL,
                     monitor='loss',
                     verbose=1,
                     save_best_only=True,
                     mode='auto')]

#################### TRAIN ####################
#Train
model.fit_generator(generator=train_generator,
                    verbose=1,
                    epochs=nEpochs,
                    callbacks=callbacks_list,
                    use_multiprocessing=False,
                    workers=4)