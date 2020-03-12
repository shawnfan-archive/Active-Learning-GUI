# Class for data generation
# All data will not fit in the memory
# Inspired from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html

import numpy as np
import os
import keras
from keras.utils import Sequence, to_categorical
# from scipy.ndimage import zoom
from skimage.transform import resize
# import ni_utils
import nibabel as nib
import pandas as pd

import math
# import transformations as tr

HIST_FEAT_SIZE = 30


# Standard data generator class
class DataGenerator(Sequence):
    'Generates data for Keras'

    def __init__(self,
                 list_IDs,
                 labels,
                 filepath,
                 data_dir,
                 xls_filepath,
                 network_type,
                 normalize=False,
                 batch_size=4,
                 dim=(91, 109, 20),
                 n_channels=1,
                 n_classes=3,
                 shuffle=True):
        'Initialization'
        # print(resample_size)
        self.filepath = filepath
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        # self.rs=resample_size
        self.norm = normalize
        self.data_dir = data_dir
        self.dim = dim
        self.xls_filepath = xls_filepath
        self.network_type = network_type
        self.on_epoch_end()

    def __len__(self):
        'Compute # batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        [ses01, ses02], y = self.__data_generation(list_IDs_temp)

        return [ses01, ses02], y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        ses01 = np.empty((self.batch_size, *self.dim, self.n_channels))
        ses02 = np.empty((self.batch_size, *self.dim, self.n_channels))

        if self.network_type == 'classification':
            y = np.empty((self.batch_size), dtype=int)
        elif self.network_type == 'regression':
            y = np.empty((self.batch_size))
            # Load xls file
            # patFr = ni_utils.loadSubjGt(self.xls_filepath)

        # Generate data
        for i, patID in enumerate(list_IDs_temp):

            # load file
            FILENAME = self.filepath + '_' + patID + '.npy'
            FILEPATH = os.path.join(self.data_dir, FILENAME)
            brainDict = np.load(FILEPATH)

            # Get hemispheres and resample
            leftBrain = brainDict.item().get('leftBrain')
            # Resample only if specified
            # leftBrain = resize(leftBrain.astype(int), self.rs, anti_aliasing=True, mode='reflect', preserve_range=True)

            rightBrain = brainDict.item().get('rightBrain')
            # Resample only if specified
            # rightBrain = resize(rightBrain.astype(int), self.rs, anti_aliasing=True, mode='reflect', preserve_range=True)

            # normalize
            if self.norm == True:
                leftBrain = (leftBrain -
                             np.mean(leftBrain)) / np.std(leftBrain)
                rightBrain = (rightBrain -
                              np.mean(rightBrain)) / np.std(rightBrain)

            # Store sample
            ses01[i, ] = np.expand_dims(leftBrain, axis=3)
            ses02[i, ] = np.expand_dims(rightBrain, axis=3)

            if self.network_type == 'classification':
                # Store class
                y[i] = brainDict.item().get('isStroke')
            elif self.network_type == 'regression':
                # Store stroke volume
                y[i] = patFr.loc[patFr['patID'] == patID,
                                 'Stroke Volume'].iloc[0]

        return [ses01, ses02], y


# Data Generator class with histogram difference features
class DataGeneratorWithHistFeat(Sequence):
    'Generates data for Keras'

    def __init__(self,
                 list_IDs,
                 labels,
                 filepath,
                 dataDir,
                 histFeatFilepath,
                 normalize=False,
                 batch_size=4,
                 dim=(29, 73, 20),
                 n_channels=1,
                 n_classes=2,
                 shuffle=True):
        'Initialization'
        # print(resample_size)
        self.filepath = filepath
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        # self.rs=resample_size
        self.norm = normalize
        self.data_dir = dataDir
        self.dim = dim
        self.histFeatDict = np.load(histFeatFilepath)
        self.on_epoch_end()

    def __len__(self):
        'Compute # batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        [ses01, ses02, histFeatMat], y = self.__data_generation(list_IDs_temp)

        return [ses01, ses02, histFeatMat], y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        ses01 = np.empty((self.batch_size, *self.dim, self.n_channels))
        ses02 = np.empty((self.batch_size, *self.dim, self.n_channels))
        histFeatMat = np.empty((self.batch_size, HIST_FEAT_SIZE))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, patID in enumerate(list_IDs_temp):

            # load file
            FILENAME = self.filepath + '_' + patID + '.npy'
            FILEPATH = os.path.join(self.data_dir, FILENAME)
            brainDict = np.load(FILEPATH)

            # Get hemispheres and resample
            leftBrain = brainDict.item().get('leftBrain')
            # Resample only if specified
            # leftBrain = resize(leftBrain.astype(int), self.rs, anti_aliasing=True, mode='reflect', preserve_range=True)

            rightBrain = brainDict.item().get('rightBrain')
            # Resample only if specified
            # rightBrain = resize(rightBrain.astype(int), self.rs, anti_aliasing=True, mode='reflect', preserve_range=True)

            # normalize
            if self.norm == True:
                leftBrain = (leftBrain -
                             np.mean(leftBrain)) / np.std(leftBrain)
                rightBrain = (rightBrain -
                              np.mean(rightBrain)) / np.std(rightBrain)

            # Get histogram feature
            histFeat = self.histFeatDict[patID]

            # Store sample
            ses01[i, ] = np.expand_dims(leftBrain, axis=3)
            ses02[i, ] = np.expand_dims(rightBrain, axis=3)
            histFeatMat[i, ] = histFeat

            # Store class
            y[i] = brainDict.item().get('isStroke')

        return [ses01, ses02, histFeatMat], y


### STROKE
class DataGenerator_stroke(Sequence):
    'Generates data for Keras'

    def __init__(self,
                 list_IDs,
                 labels,
                 filepath,
                 data_dir,
                 xls_filepath,
                 normalize=False,
                 batch_size=4,
                 dim=(91, 109, 20),
                 n_channels=1,
                 n_classes=2,
                 shuffle=True):
        'Initialization'
        # print(resample_size)
        self.filepath = filepath
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        # self.rs=resample_size
        self.norm = normalize
        self.data_dir = data_dir
        self.dim = dim
        self.xls_filepath = xls_filepath
        self.on_epoch_end()

    def __len__(self):
        'Compute # batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        left, y = self.__data_generation(list_IDs_temp)

        return left, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        ses01 = np.empty((self.batch_size, *self.dim))
        ses02 = np.empty((self.batch_size, *self.dim, self.n_channels))

        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, patID in enumerate(list_IDs_temp):

            server = '/nfsshare'
            #             server = '/collab'

            #Open data dataframe
            data_df = pd.read_csv(
                server +
                '/gianca-group/dpena3/active_learning/ATLAS_feb2020/pre-processing/ATLAS_stroke_labels_20200122.csv'
            )
            FILEPATH = server + data_df[data_df['filename'] ==
                                        patID]['brain_path'].values[0]
            sliceNum = data_df[data_df['filename'] ==
                               patID]['slice_num'].values[0]

            # load file
            #             FILENAME = str(patID) + '.npy'
            #             FILEPATH = os.path.join(self.data_dir, FILENAME)
            #             brain = np.load(FILEPATH, allow_pickle=True)

            #Load in nifti, grab object, go to chosen slice, and clip files
            img = nib.load(FILEPATH)
            data = np.array(img.dataobj)
            brain = (data[:, :, sliceNum])
            brain = resize(brain, (192, 192),
                           anti_aliasing=True,
                           mode='reflect',
                           preserve_range=True)

            #             data = np.clip(data, 0, 100)

            # Store sample
            #             ses01[i,] = np.expand_dims(brain, axis=3)
            ses01[i, ] = np.expand_dims(brain, axis=2)

            # Store class
            y[i] = self.labels[patID]

        return ses01, to_categorical(y, num_classes=self.n_classes)


class DataGenerator_stroke_regression(Sequence):
    'Generates data for Keras'

    def __init__(self,
                 list_IDs,
                 labels,
                 filepath,
                 data_dir,
                 xls_filepath,
                 normalize=False,
                 batch_size=4,
                 dim=(91, 109, 20),
                 n_channels=1,
                 n_classes=2,
                 shuffle=True):
        'Initialization'
        # print(resample_size)
        self.filepath = filepath
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        # self.rs=resample_size
        self.norm = normalize
        self.data_dir = data_dir
        self.dim = dim
        self.xls_filepath = xls_filepath
        self.on_epoch_end()

    def __len__(self):
        'Compute # batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        left, y = self.__data_generation(list_IDs_temp)

        return left, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        ses01 = np.empty((self.batch_size, *self.dim))
        ses02 = np.empty((self.batch_size, *self.dim, self.n_channels))

        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, patID in enumerate(list_IDs_temp):

            server = '/nfsshare'
            #             server = '/collab'

            #Open data dataframe
            data_df = pd.read_csv(
                server +
                '/gianca-group/dpena3/active_learning/ATLAS_feb2020/pre-processing/ATLAS_stroke_labels_20200122.csv'
            )
            FILEPATH = server + data_df[data_df['filename'] ==
                                        patID]['brain_path'].values[0]
            sliceNum = data_df[data_df['filename'] ==
                               patID]['slice_num'].values[0]

            #Load in nifti, grab object, go to chosen slice, and clip files
            img = nib.load(FILEPATH)
            data = np.array(img.dataobj)
            brain = (data[:, :, sliceNum])
            brain = resize(brain, (192, 192),
                           anti_aliasing=True,
                           mode='reflect',
                           preserve_range=True)

            #             data = np.clip(data, 0, 100)

            # Store sample
            #             ses01[i,] = np.expand_dims(brain, axis=3)
            ses01[i, ] = np.expand_dims(brain, axis=2)

            # Store class
            y[i] = self.labels[patID]

        return ses01, y


class DataGenerator_stroke_quartile_classification(Sequence):
    'Generates data for Keras'

    def __init__(self,
                 list_IDs,
                 labels,
                 filepath,
                 data_dir,
                 xls_filepath,
                 normalize=False,
                 batch_size=4,
                 dim=(91, 109, 20),
                 n_channels=1,
                 n_classes=2,
                 shuffle=True):
        'Initialization'
        # print(resample_size)
        self.filepath = filepath
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        # self.rs=resample_size
        self.norm = normalize
        self.data_dir = data_dir
        self.dim = dim
        self.xls_filepath = xls_filepath
        self.on_epoch_end()

    def __len__(self):
        'Compute # batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        left, y = self.__data_generation(list_IDs_temp)

        return left, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        ses01 = np.empty((self.batch_size, *self.dim))
        ses02 = np.empty((self.batch_size, *self.dim, self.n_channels))

        y = np.empty((self.batch_size, 4), dtype=int)

        # Generate data
        for i, patID in enumerate(list_IDs_temp):

            server = '/nfsshare'
            #             server = '/collab'

            #Open data dataframe
            data_df = pd.read_csv(
                server +
                '/gianca-group/dpena3/active_learning/ATLAS_feb2020/pre-processing/ATLAS_stroke_labels_20200122.csv'
            )
            FILEPATH = server + data_df[data_df['filename'] ==
                                        patID]['brain_path'].values[0]
            sliceNum = data_df[data_df['filename'] ==
                               patID]['slice_num'].values[0]

            #Load in nifti, grab object, go to chosen slice, and clip files
            img = nib.load(FILEPATH)
            data = np.array(img.dataobj)
            brain = (data[:, :, sliceNum])
            brain = resize(brain, (184, 232),
                           anti_aliasing=True,
                           mode='reflect',
                           preserve_range=True)

            #             data = np.clip(data, 0, 100)

            # Store sample
            #             ses01[i,] = np.expand_dims(brain, axis=3)
            ses01[i, ] = np.expand_dims(brain, axis=2)

            # Store class
            y[i] = self.labels[patID]

        return ses01, y


class DataGenerator_stroke_unet(Sequence):
    'Generates data for Keras'

    def __init__(self,
                 list_IDs,
                 filepath,
                 data_dir,
                 xls_filepath,
                 normalize=False,
                 batch_size=4,
                 dim=(91, 109, 20),
                 n_channels=1,
                 n_classes=2,
                 shuffle=True):
        'Initialization'
        # print(resample_size)
        self.filepath = filepath
        self.batch_size = batch_size
        #         self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        # self.rs=resample_size
        self.norm = normalize
        self.data_dir = data_dir
        self.dim = dim
        self.xls_filepath = xls_filepath
        self.on_epoch_end()

    def __len__(self):
        'Compute # batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        brains, lesions = self.__data_generation(list_IDs_temp)

        return brains, lesions

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        brains = np.empty((self.batch_size, *self.dim))
        lesions = np.empty((self.batch_size, *self.dim))
        ses01 = np.empty((self.batch_size, *self.dim))
        ses02 = np.empty((self.batch_size, *self.dim, self.n_channels))

        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, patID in enumerate(list_IDs_temp):

            server = '/nfsshare'
            #             server = '/collab'

            #File path must be relative to current working directory – app.py !!!
            brain_path = 'data/train/image/' + patID
            lesion_path = 'data/train/label/' + patID
            #CHANGE DATA TO NUMPY ARRAYS

            # sliceNum = data_df[data_df['filename'] == patID]['slice_num'].values[0]

            # load file
            #             FILENAME = str(patID) + '.npy'
            #             FILEPATH = os.path.join(self.data_dir, FILENAME)
            #             brain = np.load(FILEPATH, allow_pickle=True)
            #             resize_shape = (184, 232)
            resize_shape = (192, 192, 1)
            #             resize_shape = (256, 256)

            #Load in nifti, grab object, go to chosen slice, and clip files
            # img = nib.load(brain_path)
            # data = np.array(img.dataobj)
            # brain = (data[:,:, sliceNum-2:sliceNum+2])
            brain = np.load(brain_path, allow_pickle=True)
            brain = resize(brain,
                           resize_shape,
                           anti_aliasing=True,
                           mode='reflect',
                           preserve_range=True)
            brain = brain / np.max(brain)  #normalize brain
            brain = np.nan_to_num(brain)  #null values to 0

            #Load in nifti, grab object, go to chosen slice, and clip files
            resize_shape = (192, 192)
            # img = nib.load(lesion_path)
            # data = np.array(img.dataobj)
            # lesion = (data[:,:,sliceNum])
            lesion = np.load(lesion_path, allow_pickle=True)
            lesion = resize(lesion,
                            resize_shape,
                            anti_aliasing=True,
                            mode='reflect',
                            preserve_range=True)
            #             lesion = np.clip(lesion, 0, 1e-5)
            #             lesion = lesion / np.max(lesion)
            #             lesion = np.nan_to_num(lesion)
            #             data = np.clip(data, 0, 100)
            lesion = np.nan_to_num(lesion)

            #Masking
            lesion = lesion / 255
            lesion[lesion >= 0.5] = 1
            lesion[lesion < 0.5] = 0

            # Store sample
            #             brains[i,] = np.expand_dims(brain, axis=4)
            brains[i, ] = brain
            lesions[i, ] = np.expand_dims(lesion, axis=2)  #may need to change this to a 1 or a 3

            # Store class


#             y[i] = self.labels[patID]

        return brains, lesions


class DataGenerator_stroke_d_unet(Sequence):
    'Generates data for Keras'

    def __init__(self,
                 list_IDs,
                 filepath,
                 data_dir,
                 xls_filepath,
                 normalize=False,
                 batch_size=4,
                 dim=(91, 109, 20),
                 n_channels=1,
                 n_classes=2,
                 shuffle=True):
        'Initialization'
        # print(resample_size)
        self.filepath = filepath
        self.batch_size = batch_size
        #         self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        # self.rs=resample_size
        self.norm = normalize
        self.data_dir = data_dir
        self.dim = dim
        self.xls_filepath = xls_filepath
        self.on_epoch_end()

    def __len__(self):
        'Compute # batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        brains, lesions = self.__data_generation(list_IDs_temp)

        return brains, lesions

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        brains = np.empty((self.batch_size, 192, 192, 4))
        lesions = np.empty((self.batch_size, 192, 192, 1))

        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, patID in enumerate(list_IDs_temp):

            server = '/nfsshare'
            #             server = '/collab'

            #Open data dataframe
            # data_df = pd.read_csv(server + '/gianca-group/dpena3/active_learning/ATLAS_feb2020/pre-processing/ATLAS_stroke_labels_20200122.csv')

            brain_path = '/data/train/image/' + patID
            lesion_path = '/data/train/label/' + patID
            #CHANGE DATA TO NUMPY ARRAYS

            # sliceNum = data_df[data_df['filename'] == patID]['slice_num'].values[0]

            # load file
            #             FILENAME = str(patID) + '.npy'
            #             FILEPATH = os.path.join(self.data_dir, FILENAME)
            #             brain = np.load(FILEPATH, allow_pickle=True)
            #             resize_shape = (184, 232)
            resize_shape = (192, 192)
            #             resize_shape = (256, 256)

            #Load in nifti, grab object, go to chosen slice, and clip files
            # img = nib.load(brain_path)
            # data = np.array(img.dataobj)
            # brain = (data[:,:, sliceNum-2:sliceNum+2])
            brain = np.load(brain_path, allow_pickle=True)
            brain = resize(brain,
                           resize_shape,
                           anti_aliasing=True,
                           mode='reflect',
                           preserve_range=True)
            brain = brain / np.max(brain)  #normalize brain
            brain = np.nan_to_num(brain)  #null values to 0

            #Load in nifti, grab object, go to chosen slice, and clip files
            resize_shape = (192, 192)
            # img = nib.load(lesion_path)
            # data = np.array(img.dataobj)
            # lesion = (data[:,:,sliceNum])
            lesion = np.load(lesion_path, allow_pickle=True)
            lesion = resize(lesion,
                            resize_shape,
                            anti_aliasing=True,
                            mode='reflect',
                            preserve_range=True)
            #             lesion = np.clip(lesion, 0, 1e-5)
            #             lesion = lesion / np.max(lesion)
            #             lesion = np.nan_to_num(lesion)
            #             data = np.clip(data, 0, 100)
            lesion = np.nan_to_num(lesion)

            #Masking
            lesion = lesion / 255
            lesion[lesion >= 0.5] = 1
            lesion[lesion < 0.5] = 0

            # Store sample
            #             brains[i,] = np.expand_dims(brain, axis=4)
            brains[i, ] = brain
            lesions[i, ] = np.expand_dims(
                lesion, axis=2)  #may need to change this to a 1 or a 3

            # Store class
#             y[i] = self.labels[patID]

        return brains, lesions


####CHANGED FOR PPMI/ADNI DATASET
# Data generator class for multiclass classification
class DataGeneratorMulticlass(Sequence):
    'Generates data for Keras'

    def __init__(self,
                 list_IDs,
                 labels,
                 filepath,
                 data_dir,
                 xls_filepath,
                 normalize=False,
                 batch_size=4,
                 dim=(91, 109, 20),
                 n_channels=1,
                 n_classes=2,
                 shuffle=True):
        'Initialization'
        # print(resample_size)
        self.filepath = filepath
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        # self.rs=resample_size
        self.norm = normalize
        self.data_dir = data_dir
        self.dim = dim
        self.xls_filepath = xls_filepath
        self.on_epoch_end()

    def __len__(self):
        'Compute # batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        [left, right], y = self.__data_generation(list_IDs_temp)

        return [left, right], y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        ses01 = np.empty((self.batch_size, *self.dim, self.n_channels))
        ses02 = np.empty((self.batch_size, *self.dim, self.n_channels))

        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, patID in enumerate(list_IDs_temp):

            # load file
            FILENAME = str(patID) + '.npy'
            FILEPATH = os.path.join(self.data_dir, FILENAME)
            brainDict = np.load(FILEPATH, allow_pickle=True)

            # Get hemispheres and resample
            ses01_brain = brainDict.item().get('ses01_brain')
            # Resample only if specified
            # leftBrain = resize(leftBrain.astype(int), self.rs, anti_aliasing=True, mode='reflect', preserve_range=True)

            ses02_brain = brainDict.item().get('ses02_brain')
            # Resample only if specified
            # rightBrain = resize(rightBrain.astype(int), self.rs, anti_aliasing=True, mode='reflect', preserve_range=True)

            # normalize
            if self.norm == True:
                ses01_brain = (ses01_brain -
                               np.mean(ses01_brain)) / np.std(ses01_brain)
                ses02_brain = (ses02_brain -
                               np.mean(ses02_brain)) / np.std(ses02_brain)

            # Store sample
            ses01[i, ] = np.expand_dims(ses01_brain, axis=3)
            ses02[i, ] = np.expand_dims(ses02_brain, axis=3)

            # Store class
            y[i] = self.labels[patID]

        return [ses01, ses02], to_categorical(y, num_classes=self.n_classes)


class DataGeneratorMulticlass_mci_increment(Sequence):
    'Generates data for Keras'

    def __init__(self,
                 list_IDs,
                 labels,
                 filepath,
                 data_dir,
                 xls_filepath,
                 normalize=False,
                 batch_size=4,
                 dim=(91, 109, 20),
                 n_channels=1,
                 n_classes=2,
                 shuffle=True):
        'Initialization'
        # print(resample_size)
        self.filepath = filepath
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        # self.rs=resample_size
        self.norm = normalize
        self.data_dir = data_dir
        self.dim = dim
        self.xls_filepath = xls_filepath
        self.on_epoch_end()

    def __len__(self):
        'Compute # batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        [left, right], y = self.__data_generation(list_IDs_temp)

        return [left, right], y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        ses01 = np.empty((self.batch_size, *self.dim, self.n_channels))
        ses02 = np.empty((self.batch_size, *self.dim, self.n_channels))

        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, patID in enumerate(list_IDs_temp):

            ses01_brain, ses02_brain = increment_mci(patID)

            # normalize
            if self.norm == True:
                ses01_brain = (ses01_brain -
                               np.mean(ses01_brain)) / np.std(ses01_brain)
                ses02_brain = (ses02_brain -
                               np.mean(ses02_brain)) / np.std(ses02_brain)

            # Store sample
            ses01[i, ] = np.expand_dims(ses01_brain, axis=3)
            ses02[i, ] = np.expand_dims(ses02_brain, axis=3)

            # Store class
            y[i] = self.labels[patID]

        return [ses01, ses02], to_categorical(y, num_classes=self.n_classes)


class DataGeneratorMulticlass_mci_decrement(Sequence):
    'Generates data for Keras'

    def __init__(self,
                 list_IDs,
                 labels,
                 filepath,
                 data_dir,
                 xls_filepath,
                 normalize=False,
                 batch_size=4,
                 dim=(91, 109, 20),
                 n_channels=1,
                 n_classes=2,
                 shuffle=True):
        'Initialization'
        # print(resample_size)
        self.filepath = filepath
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        # self.rs=resample_size
        self.norm = normalize
        self.data_dir = data_dir
        self.dim = dim
        self.xls_filepath = xls_filepath
        self.on_epoch_end()

    def __len__(self):
        'Compute # batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        [left, right], y = self.__data_generation(list_IDs_temp)

        return [left, right], y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        ses01 = np.empty((self.batch_size, *self.dim, self.n_channels))
        ses02 = np.empty((self.batch_size, *self.dim, self.n_channels))

        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, patID in enumerate(list_IDs_temp):

            ses01_brain, ses02_brain = decrement_mci(patID)

            # normalize
            if self.norm == True:
                ses01_brain = (ses01_brain -
                               np.mean(ses01_brain)) / np.std(ses01_brain)
                ses02_brain = (ses02_brain -
                               np.mean(ses02_brain)) / np.std(ses02_brain)

            # Store sample
            ses01[i, ] = np.expand_dims(ses01_brain, axis=3)
            ses02[i, ] = np.expand_dims(ses02_brain, axis=3)

            # Store class
            y[i] = self.labels[patID]

        return [ses01, ses02], to_categorical(y, num_classes=self.n_classes)


def pad_number(number):

    if number < 10:
        return '0' + str(number)
    else:
        return str(number)


class DataGeneratorMulticlass_mci_3timepoints(Sequence):
    'Generates data for Keras'

    def __init__(self,
                 list_IDs,
                 labels,
                 filepath,
                 data_dir,
                 xls_filepath,
                 normalize=False,
                 batch_size=4,
                 dim=(91, 109, 20),
                 n_channels=1,
                 n_classes=2,
                 shuffle=True):
        'Initialization'
        # print(resample_size)
        self.filepath = filepath
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        # self.rs=resample_size
        self.norm = normalize
        self.data_dir = data_dir
        self.dim = dim
        self.xls_filepath = xls_filepath
        self.on_epoch_end()

    def __len__(self):
        'Compute # batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        [first, second, third], y = self.__data_generation(list_IDs_temp)

        return [first, second, third], y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        ses01 = np.empty((self.batch_size, *self.dim, self.n_channels))
        ses02 = np.empty((self.batch_size, *self.dim, self.n_channels))
        ses03 = np.empty((self.batch_size, *self.dim, self.n_channels))

        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, patID in enumerate(list_IDs_temp):

            path_to_mci = '/collab/gianca-group/dpena3/ADNI_MCI/pre-processing/data_model-ready/64x80x64_20191022/'

            ############# FIRST SESSION ###################

            #Loop to grab data for both sessions
            passed_first = False

            #Loop through sessions 1 through 20
            for first_session in range(1, 20):

                #If still haven't found a first session
                if not passed_first:

                    #Check if this session image exists
                    if os.path.exists(path_to_mci + str(patID) + '_ses-' +
                                      pad_number(first_session) +
                                      '_64x80x64.npy'):

                        #Load image into second session
                        ses01_brain = np.load(
                            path_to_mci + str(patID) + '_ses-' +
                            pad_number(first_session) + '_64x80x64.npy',
                            allow_pickle=True)
                        final_first_session = first_session
                        passed_first = True


#             print(final_first_session)

############# SECOND SESSION ###################
#Repeat for second session
            passed_second = False

            #Increment
            for second_session in range(final_first_session + 1, 20):

                #If still haven't found a first session
                if not passed_second:

                    #Check if this session image exists
                    if os.path.exists(path_to_mci + str(patID) + '_ses-' +
                                      pad_number(second_session) +
                                      '_64x80x64.npy'):

                        #Load image into second session
                        ses02_brain = np.load(
                            path_to_mci + str(patID) + '_ses-' +
                            pad_number(second_session) + '_64x80x64.npy',
                            allow_pickle=True)
                        final_second_session = second_session
                        passed_second = True

            ############# THIRD SESSION ###################
            #Repeat for second session
            passed_third = False

            #Increment
            for third_session in range(final_second_session + 1, 20):

                #If still haven't found a first session
                if not passed_third:

                    #Check if this session image exists
                    if os.path.exists(path_to_mci + str(patID) + '_ses-' +
                                      pad_number(third_session) +
                                      '_64x80x64.npy'):

                        #Load image into second session
                        ses03_brain = np.load(
                            path_to_mci + str(patID) + '_ses-' +
                            pad_number(third_session) + '_64x80x64.npy',
                            allow_pickle=True)
                        passed_third = True

            #If haven't found third session
            if not passed_third:
                ses03_brain = np.full(ses02_brain.shape, 0)

            # normalize
            if self.norm == True:
                ses01_brain = (ses01_brain -
                               np.mean(ses01_brain)) / np.std(ses01_brain)
                ses02_brain = (ses02_brain -
                               np.mean(ses02_brain)) / np.std(ses02_brain)

            # Store sample
            ses01[i, ] = np.expand_dims(ses01_brain, axis=3)
            ses02[i, ] = np.expand_dims(ses02_brain, axis=3)
            ses03[i, ] = np.expand_dims(ses03_brain, axis=3)

            # Store class
            y[i] = self.labels[patID]

        return [ses01, ses02,
                ses03], to_categorical(y, num_classes=self.n_classes)


class DataGeneratorMulticlass_zeroPad(Sequence):
    'Generates data for Keras'

    def __init__(self,
                 list_IDs,
                 labels,
                 filepath,
                 data_dir,
                 xls_filepath,
                 normalize=False,
                 batch_size=4,
                 dim=(91, 109, 20),
                 n_channels=1,
                 n_classes=2,
                 shuffle=True):
        'Initialization'
        # print(resample_size)
        self.filepath = filepath
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        # self.rs=resample_size
        self.norm = normalize
        self.data_dir = data_dir
        self.dim = dim
        self.xls_filepath = xls_filepath
        self.on_epoch_end()

    def __len__(self):
        'Compute # batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        [left, right], y = self.__data_generation(list_IDs_temp)

        return [left, right], y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        ses01 = np.empty((self.batch_size, *self.dim, self.n_channels))
        ses02 = np.empty((self.batch_size, *self.dim, self.n_channels))

        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, patID in enumerate(list_IDs_temp):

            # load file
            FILENAME = str(patID) + '.npy'
            FILEPATH = os.path.join(self.data_dir, FILENAME)
            brainDict = np.load(FILEPATH, allow_pickle=True)

            # Get hemispheres and resample
            ses01_brain = brainDict.item().get('ses01_brain')
            # Resample only if specified
            # leftBrain = resize(leftBrain.astype(int), self.rs, anti_aliasing=True, mode='reflect', preserve_range=True)

            ses02_brain = brainDict.item().get('ses02_brain')
            # Resample only if specified
            # rightBrain = resize(rightBrain.astype(int), self.rs, anti_aliasing=True, mode='reflect', preserve_range=True)

            ######## ZERO PADDING ######

            #X% chance of zero padding
            init_array = np.concatenate((np.ones(10), np.zeros(90)))
            random_choice = int(np.random.choice(init_array))

            #If chose padding, set one of the brains to zero
            if random_choice == 1:
                second_random_choice = int(np.random.choice([0, 1]))

                if second_random_choice == 0:
                    ses01_brain = np.full(ses01_brain.shape, 0)
                else:
                    ses02_brain = np.full(ses02_brain.shape, 0)

            ##########################

            # normalize
            if self.norm == True:
                ses01_brain = (ses01_brain -
                               np.mean(ses01_brain)) / np.std(ses01_brain)
                ses02_brain = (ses02_brain -
                               np.mean(ses02_brain)) / np.std(ses02_brain)

            # Store sample
            ses01[i, ] = np.expand_dims(ses01_brain, axis=3)
            ses02[i, ] = np.expand_dims(ses02_brain, axis=3)

            # Store class
            y[i] = self.labels[patID]

        return [ses01, ses02], to_categorical(y, num_classes=self.n_classes)


# class DataGeneratorMulticlass_dataAug(Sequence):
#     'Generates data for Keras'
#     def __init__(self, list_IDs, labels, filepath, data_dir, xls_filepath, normalize=False, batch_size=4, dim=(91, 109, 20), n_channels=1, n_classes=2, shuffle=True):
#         'Initialization'
#         # print(resample_size)
#         self.filepath = filepath
#         self.batch_size = batch_size
#         self.labels = labels
#         self.list_IDs = list_IDs
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.shuffle = shuffle
#         # self.rs=resample_size
#         self.norm = normalize
#         self.data_dir = data_dir
#         self.dim = dim
#         self.xls_filepath = xls_filepath
#         self.on_epoch_end()

#     def __len__(self):
#         'Compute # batches per epoch'
#         return int(np.floor(len(self.list_IDs)/self.batch_size))

#     def __getitem__(self, index):
#         'Generate one batch of data'
#         # Generate indexes of the batch
#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

#         # Find list of IDs
#         list_IDs_temp = [self.list_IDs[k] for k in indexes]

#         # Generate data
#         [left, right], y = self.__data_generation(list_IDs_temp)

#         return [left, right], y

#     def on_epoch_end(self):
#         'Updates indexes after each epoch'
#         self.indexes = np.arange(len(self.list_IDs))
#         if self.shuffle == True:
#             np.random.shuffle(self.indexes)

#     def __data_generation(self, list_IDs_temp):
#         'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
#         # Initialization
#         ses01 = np.empty((self.batch_size, *self.dim, self.n_channels))
#         ses02 = np.empty((self.batch_size, *self.dim, self.n_channels))

#         y = np.empty((self.batch_size), dtype=int)

#         # Generate data
#         for i, patID in enumerate(list_IDs_temp):

#             # load file
#             FILENAME = str(patID) + '.npy'
#             FILEPATH = os.path.join(self.data_dir, FILENAME)
#             brainDict = np.load(FILEPATH, allow_pickle=True)

#             # Get hemispheres and resample
#             ses01_brain = brainDict.item().get('ses01_brain')
#             # Resample only if specified
#             # leftBrain = resize(leftBrain.astype(int), self.rs, anti_aliasing=True, mode='reflect', preserve_range=True)

#             ses02_brain = brainDict.item().get('ses02_brain')
#             # Resample only if specified
#             # rightBrain = resize(rightBrain.astype(int), self.rs, anti_aliasing=True, mode='reflect', preserve_range=True)

#             ######## DATA AUGMENTATION ######

#             #10% chance of augmentation
#             init_array = np.concatenate((np.ones(10), np.zeros(90)))
#             random_choice = int(np.random.choice(init_array))

#             if random_choice == 1:

#                 #Load ses01 nii
#                 FOLDER_NIFTI = '/collab/dpena3/ADNI_longitudinal_MRI/ADNI_fs/network_data/cropped_empty_nifti'
#                 FILENAME_SES01 = patID + '_ses01.nii.gz'

#                 imgNi = nil_img.load_img(os.path.join(FOLDER_NIFTI, FILENAME_SES01))

#                 #If data aug, do some random operations (rotation, translation, scaling) to both
#                 # rotation around center
#                 t = np.array(ses01_brain.shape)/2.
#                 trMat1 = tr.compose_matrix( translate=-1*t )
#                 trMat2 = tr.compose_matrix( angles=np.random.uniform(0,.1,3)*math.pi) # example angles=[0,0.3*math.pi,0]
#                 trMat3 = tr.compose_matrix(translate=1*t )
#                 # compose Rotation matrix
#                 trMatRot = np.dot(trMat2, trMat1)
#                 trMatRot = np.dot(trMat3, trMatRot)

#                 # translation
#                 trMatTr = tr.compose_matrix( translate=np.random.randint(1,3,3) )

#                 # scaling
#                 trMatSc = tr.compose_matrix( scale=np.random.uniform(0.98,1.02,3) )

#                 # combine rotation translation and scaling (in this order)
#                 trMatFin = np.dot(trMatTr, trMatRot)
#                 trMatFin = np.dot(trMatSc, trMatFin)
#                 trMatFin = np.dot(imgNi.affine, trMatFin)

#                 imgNi3 = nil_img.resample_img( imgNi, target_affine=trMatFin, target_shape=imgNi.shape, interpolation='linear' )

#                 #Get numpy and resize
#                 ses01_brain = imgNi3.get_data()

#             ##########################

#             # normalize
#             if self.norm == True:
#                 ses01_brain = (ses01_brain - np.mean(ses01_brain))/np.std(ses01_brain)
#                 ses02_brain = (ses02_brain - np.mean(ses02_brain))/np.std(ses02_brain)

#             # Store sample
#             ses01[i,] = np.expand_dims(ses01_brain, axis=3)
#             ses02[i,] = np.expand_dims(ses02_brain, axis=3)

#             # Store class
#             y[i] = self.labels[patID]

#         return [ses01, ses02], to_categorical(y, num_classes=self.n_classes)

#Code for datageneration on the fly
#          #############################Session 1
#             #Load ses01 nii
#             FOLDER_NIFTI = '/collab/dpena3/ADNI_longitudinal_MRI/ADNI_fs/network_data/cropped_empty_nifti'
#             FILENAME_SES01 = patID + '_ses01.nii.gz'

#             imgNi = nil_img.load_img(os.path.join(FOLDER_NIFTI, FILENAME_SES01))

#             #Transform ses01 nii (randomly)
#             # rotation around center
#             t = np.array(imgNi.shape)/2.
#             trMat1 = tr.compose_matrix( translate=-1*t )
#             trMat2 = tr.compose_matrix( angles=np.random.uniform(0,.1,3)*math.pi) # example angles=[0,0.3*math.pi,0]
#             trMat3 = tr.compose_matrix(translate=1*t )
#             # compose Rotation matrix
#             trMatRot = np.dot(trMat2, trMat1)
#             trMatRot = np.dot(trMat3, trMatRot)

#             # translation
#             trMatTr = tr.compose_matrix( translate=np.random.randint(1,3,3) )

#             # scaling
#             trMatSc = tr.compose_matrix( scale=np.random.uniform(0.95,1.05,3) )

#             # combine rotation translation and scaling (in this order)
#             trMatFin = np.dot(trMatTr, trMatRot)
#             trMatFin = np.dot(trMatSc, trMatFin)
#             trMatFin = np.dot(imgNi.affine, trMatFin)

#             imgNi3 = nil_img.resample_img( imgNi, target_affine=trMatFin, target_shape=imgNi.shape, interpolation='linear' )

#             #Get numpy and resize
#             ses01_brain = imgNi3.get_data()

#             #############################Session 2
#             #Load ses02 nii
#             FOLDER_NIFTI = '/collab/dpena3/ADNI_longitudinal_MRI/ADNI_fs/network_data/cropped_empty_nifti'
#             FILENAME_SES02 = patID + '_ses02.nii.gz'

#             imgNi = nil_img.load_img(os.path.join(FOLDER_NIFTI, FILENAME_SES02))

#             #Transform ses01 nii (randomly)
#             # rotation around center
#             t = np.array(imgNi.shape)/2.
#             trMat1 = tr.compose_matrix( translate=-1*t )
#             trMat2 = tr.compose_matrix( angles=np.random.uniform(0,.1,3)*math.pi) # example angles=[0,0.3*math.pi,0]
#             trMat3 = tr.compose_matrix(translate=1*t )
#             # compose Rotation matrix
#             trMatRot = np.dot(trMat2, trMat1)
#             trMatRot = np.dot(trMat3, trMatRot)

#             # translation
#             trMatTr = tr.compose_matrix( translate=np.random.randint(1,3,3) )

#             # scaling
#             trMatSc = tr.compose_matrix( scale=np.random.uniform(0.95,1.05,3) )

#             # combine rotation translation and scaling (in this order)
#             trMatFin = np.dot(trMatTr, trMatRot)
#             trMatFin = np.dot(trMatSc, trMatFin)
#             trMatFin = np.dot(imgNi.affine, trMatFin)

#             imgNi3 = nil_img.resample_img( imgNi, target_affine=trMatFin, target_shape=imgNi.shape, interpolation='linear' )

#             #Get numpy and resize
#             ses02_brain = imgNi3.get_data()

#             ###RESIZE
#             ####CROPPED IMAGE TO NETWORK DIMENSIONS
#             nPlanes = 20  # # planes in z direction
#             scale = 0.2 # scale for resizing x-y planes
#             resample_size = (int(182 * scale), int(218 * scale), nPlanes)

#             ses01_brain = resize(ses01_brain.astype(int), resample_size, mode='reflect', preserve_range=True)


def increment_mci(patID):

    path_to_mci = '/collab/gianca-group/dpena3/ADNI_MCI/pre-processing/data_model-ready/64x80x64_20191022/'

    ############# FIRST SESSION ###################

    #Loop to grab data for both sessions
    passed_first = False

    #Loop through sessions 1 through 15
    for first_session in range(1, 20):

        #If still haven't found a first session
        if not passed_first:

            #Check if this session image exists
            if os.path.exists(path_to_mci + str(patID) + '_ses-' +
                              pad_number(first_session) + '_64x80x64.npy'):

                #Load image into second session
                ses01_brain = np.load(path_to_mci + str(patID) + '_ses-' +
                                      pad_number(first_session) +
                                      '_64x80x64.npy',
                                      allow_pickle=True)
                final_first_session = first_session
                passed_first = True
    ############# SECOND SESSION ###################
    #Repeat for second session
    passed_second = False

    #Increment
    for second_session in range(final_first_session + 1, 20):

        #If still haven't found a first session
        if not passed_second:

            #Check if this session image exists
            if os.path.exists(path_to_mci + str(patID) + '_ses-' +
                              pad_number(second_session) + '_64x80x64.npy'):

                #Load image into second session
                ses02_brain = np.load(path_to_mci + str(patID) + '_ses-' +
                                      pad_number(second_session) +
                                      '_64x80x64.npy',
                                      allow_pickle=True)
                passed_second = True

    return ses01_brain, ses02_brain


def decrement_mci(patID):
    path_to_mci = '/collab/gianca-group/dpena3/ADNI_MCI/pre-processing/data_model-ready/64x80x64_20191022/'

    ############# FIRST SESSION ###################

    #Loop to grab data for both sessions
    passed_first = False

    #Loop through sessions 1 through 15
    for first_session in range(1, 20):

        #If still haven't found a first session
        if not passed_first:

            #Check if this session image exists
            if os.path.exists(path_to_mci + str(patID) + '_ses-' +
                              pad_number(first_session) + '_64x80x64.npy'):

                #Load image into second session
                ses01_brain = np.load(path_to_mci + str(patID) + '_ses-' +
                                      pad_number(first_session) +
                                      '_64x80x64.npy',
                                      allow_pickle=True)
                final_first_session = first_session
                passed_first = True
    #             print(final_first_session)

    ############# SECOND SESSION ###################
    #Repeat for second session
    passed_second = False

    #Loop through sessions 1 through 15
    #             for second_session in range(final_first_session+1, 15):

    #Decrement
    for second_session in range(20, final_first_session, -1):

        #If still haven't found a first session
        if not passed_second:

            #Check if this session image exists
            if os.path.exists(path_to_mci + str(patID) + '_ses-' +
                              pad_number(second_session) + '_64x80x64.npy'):

                #Load image into second session
                ses02_brain = np.load(path_to_mci + str(patID) + '_ses-' +
                                      pad_number(second_session) +
                                      '_64x80x64.npy',
                                      allow_pickle=True)
                passed_second = True

    return ses01_brain, ses02_brain


class DataGeneratorMulticlass_withNewMCI(Sequence):
    'Generates data for Keras'

    def __init__(self,
                 list_IDs,
                 labels,
                 filepath,
                 data_dir,
                 xls_filepath,
                 normalize=False,
                 batch_size=4,
                 dim=(91, 109, 20),
                 n_channels=1,
                 n_classes=2,
                 shuffle=True):
        'Initialization'
        # print(resample_size)
        self.filepath = filepath
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        # self.rs=resample_size
        self.norm = normalize
        self.data_dir = data_dir
        self.dim = dim
        self.xls_filepath = xls_filepath
        self.on_epoch_end()

    def __len__(self):
        'Compute # batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        [left, right], y = self.__data_generation(list_IDs_temp)

        return [left, right], y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        ses01 = np.empty((self.batch_size, *self.dim, self.n_channels))
        ses02 = np.empty((self.batch_size, *self.dim, self.n_channels))

        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, patID in enumerate(list_IDs_temp):

            if 'mci' in patID:
                ses01_brain, ses02_brain = decrement_mci(patID)

            else:

                # load file
                FILENAME = str(patID) + '.npy'
                FILEPATH = os.path.join(self.data_dir, FILENAME)
                brainDict = np.load(FILEPATH, allow_pickle=True)

                # Get hemispheres and resample
                ses01_brain = brainDict.item().get('ses01_brain')
                # Resample only if specified
                # leftBrain = resize(leftBrain.astype(int), self.rs, anti_aliasing=True, mode='reflect', preserve_range=True)

                ses02_brain = brainDict.item().get('ses02_brain')
                # Resample only if specified
                # rightBrain = resize(rightBrain.astype(int), self.rs, anti_aliasing=True, mode='reflect', preserve_range=True)

            # normalize
            if self.norm == True:
                ses01_brain = (ses01_brain -
                               np.mean(ses01_brain)) / np.std(ses01_brain)
                ses02_brain = (ses02_brain -
                               np.mean(ses02_brain)) / np.std(ses02_brain)

            # Store sample
            ses01[i, ] = np.expand_dims(ses01_brain, axis=3)
            ses02[i, ] = np.expand_dims(ses02_brain, axis=3)

            # Store class
            y[i] = self.labels[patID]

        return [ses01, ses02], to_categorical(y, num_classes=self.n_classes)