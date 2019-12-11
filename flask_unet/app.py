from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import requests
import pickle
import json
import nibabel as nib
from skimage.transform import resize
import skimage.io as io
import pandas as pd
import os
from datetime import datetime
import tensorflow as tf
from unet.model import *
from unet.data import *
import keras
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# data = np.load("/Users/shawnfan/Downloads/test1.npy")
# io.imsave(("/Users/shawnfan/Downloads/postermap.png"), data)

output_path = "/Users/shawnfan/Dropbox/active_learning_20191115/Active-Learning-GUI/flask_unet/output"

#GUI canvas dimensions
canvas_width = 436
canvas_height = 364

images = [{
    "image_id": "00001",
    "disease": "Stroke",
    "path": "resized_input"
}, {
    "image_id": "00002",
    "disease": "Stroke",
    "path": "Site6_031923___101"
}, {
    "image_id": "00002",
    "disease": "Stroke",
    "path": "Site6_031923___101 copy"
},{
    "image_id": "00003",
    "disease": "Stroke",
    "path": "Site6_031923___102"
}, {
    "image_id": "00003",
    "disease": "Stroke",
    "path": "Site6_031923___102 copy"
}, {
    "image_id": "00004",
    "disease": "Stroke",
    "path": "Site6_031923___103"
}, {
    "image_id": "00004",
    "disease": "Stroke",
    "path": "Site6_031923___103 copy"
}]

activation_map = {'filename': 'output/resized_input.png', 'activation': []}

#Load in data (nifti)
img = nib.load(
    '/Users/shawnfan/Dropbox/active_learning_20191115/Active-Learning-GUI/flask_unet/data/031923_t1w_deface_stx.nii.gz'
)
data = np.array(img.dataobj)
data = (data[:, :, 120])
data = resize(data, (256, 256), mode='reflect', preserve_range=True, order=3)


def resizeArray(data):
    data = resize(data, (canvas_height, canvas_width),
                  mode='reflect',
                  preserve_range=True,
                  order=3)
    return data


def resizeImage(data):
    # resize brain image to GUI canvas size
    data = resize(data, (canvas_height, canvas_width),
                  mode='reflect',
                  preserve_range=True,
                  order=3)
    io.imsave(("output/resized_input.png"), data / 100)
    return None

#Callback that will keep track of training times
class TimeHistory(keras.callbacks.Callback):

    current_epoch = 0

    #Once training begins...
    def on_train_begin(self, logs={}):

        #Create array
        self.times = []

        #Creates textfile we will update with time
        file = open("testfile.txt", "w")
        file.write("Training Started...\n")
        file.close()

    #Once epoch begins...
    def on_epoch_begin(self, batch, logs={}):

        self.epoch_time_start = time.time()

    #Once epoch finishes...
    def on_epoch_end(self, batch, logs={}):

        #Calculate and append elapsed time
        elapsed = (time.time() - self.epoch_time_start)
        self.times.append(elapsed)
        TimeHistory.current_epoch = TimeHistory.current_epoch + 1

        #Write to file
        with open("testfile.txt", "a") as file:
            file.write("Elapsed Time: " + str(elapsed) + "\n")
            file.close()


# # Generate resized input.png
# img = nib.load('flask_unet/data/031923_t1w_deface_stx.nii.gz')
# data = np.array(img.dataobj)
# data = (data[:,:,120])
# resizeImage(data)


def loadActivationMap(filename):
    # load activation map from numpy file
    numpy_activation = np.load(filename)

    activation_converted = []
    for row in numpy_activation[0]:
        # numpy_activation is an array with 1 element
        row_converted = []
        for val in row:
            # val is an array with 1 element
            if val[0] > 0:
                row_converted.append(1)
            else:
                row_converted.append(0)
        activation_converted.append(row_converted)

    numpy_activation = np.array(activation_converted)
    resized_numpy_activation = resizeArray(numpy_activation)

    activation = resized_numpy_activation.tolist()

    return activation


def findLatestOutput():

    files = os.listdir(output_path)
    files.remove('ex.ipynb')
    files.remove('ground_truth.png')
    files.remove('input.png')
    files.remove('resized_input.png')

    files.sort(key=lambda file: datetime.strptime(file, '%Y_%m_%d_%H_%M'))

    latest = files[-1]

    return latest


def getPrediction():
    #URL with the predict method
    url = 'http://localhost:5000/getPrediction'

    j_data = json.dumps(data.tolist())

    #Create headers for json
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}

    #Send a post request
    r = requests.post(url, data=j_data, headers=headers)

    return None


def loadCorrectedActivationMap(map, map_width, map_height):
    #Convert map - a 1D array into a 2d array
    #Convert array into numpy array and resize

    loaded_map = []

    for row_index in range(map_height):
        row = []
        for column_index in range(map_width):
            index = row_index * map_width + column_index
            row.append(map[index])
        loaded_map.append(row)

    numpy_map = np.array(loaded_map)
    numpy_map = resizeArray(numpy_map)

    return numpy_map


def updateTrainingData(corrected_map, image):
    # corrected_map: numpy array of corrected activation map

    #Current time
    now = datetime.now()
    cur_date = str(now.year) + '_' + str(now.month) + '_' + str(
        now.day) + '_' + str(now.hour) + '_' + str(now.minute)

    # save corrected activation map
    io.imsave('data/train/label/' + cur_date + image['image_id'] + '.jpeg',
              corrected_map)

    # save input image
    io.imsave('data/train/image/' + cur_date + image['image_id'] + '.jpeg',
              data / 100)

    return None


def retrain(from_scratch):

    if from_scratch:
        #URL with the predict method
        url = 'http://localhost:5000/initialModel'
    else:
        url = 'http://localhost:5000/retrain'

    #Jsonify
    # j_data = json.dumps(data.tolist())
    j_data = json.dumps('hi')

    #Create headers for json
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}

    #Send a post request
    r = requests.post(url, data=j_data, headers=headers)

    #Retrieve output and save/update model tracking csv
    print(json.loads(r.text))
    model_tracking = pd.read_csv('models/model_tracking.csv')
    model_tracking = model_tracking.append(json.loads(r.text))
    model_tracking.to_csv('models/model_tracking.csv', index=False)
    print('Model Data Saved!')

    return None


# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

#Create model
model = unet()

#This is needed to ensure you are using the correct tensor graph
graph = tf.get_default_graph()


# sanity check route
@app.route('/')
@app.route('/home')
def home():
    return 'Test'


@app.route('/active_learning', methods=['GET', 'POST'])
def active_learning():
    response_object = {'status': 'success'}
    if request.method == 'POST':
        # POST
        post_data = request.get_json()

        # from Vue: payload = { image: this.current_image, corrected_activation: corrected_activation }
        corrected_activation = post_data.get('corrected_activation')
        image = post_data.get('image')
        # convert corrected activation map to NumPy array
        corrected_activation = loadCorrectedActivationMap(
            corrected_activation, canvas_width, canvas_height)

        # add corrected map and corresponding image to train folder
        updateTrainingData(corrected_activation, image)

        retrain(True)

        getPrediction()

        response_object['message'] = 'Progress saved!'
    else:
        # GET
        # find latest output folder and prediction.py
        latest_output = findLatestOutput()
        activation_map['activation'] = loadActivationMap(output_path + '/' +
                                                         latest_output +
                                                         '/prediction.npy')
        # ActivationMaps['canvas_width'] = len(activation[0])
        # ActivationMaps['canvas_height'] = len(activation)
        response_object['activation_map'] = activation_map
        response_object['images'] = images
    return jsonify(response_object)


#API page
@app.route('/getPrediction', methods=['POST'])
#Prediction method
def makecalc():

    global graph
    with graph.as_default():

        #Current time
        now = datetime.now()
        cur_date = str(now.year) + '_' + str(now.month) + '_' + str(
            now.day) + '_' + str(now.hour) + '_' + str(now.minute)

        if not os.path.exists('output/' + cur_date):
            os.mkdir('output/' + cur_date)

        #Retrieve request
        data = request.get_json()

        #Load in latest iteration of model
        df = pd.read_csv('models/model_tracking.csv', parse_dates=['date'])
        latest_model = df.sort_values('date',
                                      ascending=False).iloc[0]['model_name']
        # latest_model = 'unet_stroke_20191108.hdf5'
        model.load_weights('models/' + latest_model)

        #Predict on data and save image
        pred = model.predict(np.array(data).reshape(1, 256, 256, 1))
        io.imsave(("output/" + cur_date + '/' + "prediction.png"),
                  pred[0, :, :, 0])

        #Return max value
        max_value = np.array2string(pred.max())

        #Save as numpy arrays
        np.save('output/' + cur_date + '/' + 'input.npy', np.array(data))
        np.save('output/' + cur_date + '/' + 'prediction.npy', np.array(pred))

        #Return result
        return jsonify(max_value)


#Retrain page
@app.route('/retrain', methods=['POST'])
#Prediction method
def retrain_model():

    global graph
    with graph.as_default():

        #Load in latest iteration of model
        df = pd.read_csv('models/model_tracking.csv', parse_dates=['date'])
        latest_model = df.sort_values('date',
                                      ascending=False).iloc[0]['model_name']
        model.load_weights('models/' + latest_model)

        #Re-train parameters
        data_gen_args = dict(rotation_range=0.2,
                             width_shift_range=0.05,
                             height_shift_range=0.05,
                             shear_range=0.05,
                             zoom_range=0.05,
                             horizontal_flip=True,
                             fill_mode='nearest')

        #Train generator
        myGene = trainGenerator(1,
                                'data/train',
                                'image',
                                'label',
                                data_gen_args,
                                save_to_dir=None)

        #Check point
        now = datetime.now()
        cur_date = str(now.year) + '_' + str(now.month) + '_' + str(
            now.day) + '_' + str(now.hour) + '_' + str(now.minute)
        model_checkpoint = ModelCheckpoint('models/unet_stroke_' + cur_date +
                                           '.hdf5',
                                           monitor='loss',
                                           verbose=1,
                                           save_best_only=True)
        time_callback = TimeHistory()

        #Train model
        model.fit_generator(myGene,
                            steps_per_epoch=5,
                            epochs=25,
                            callbacks=[model_checkpoint, time_callback])

        #Data that we will send back
        data = [{
            'date': now,
            'model_name': 'unet_stroke_' + cur_date + '.hdf5',
            'from_scratch': False
        }]

        #Return result
        return jsonify(data)


#Train model from scratch
@app.route('/initialModel', methods=['POST'])
#Prediction method
def train_from_scratch():

    global graph
    with graph.as_default():

        #Re-train parameters
        data_gen_args = dict(rotation_range=0.2,
                             width_shift_range=0.05,
                             height_shift_range=0.05,
                             shear_range=0.05,
                             zoom_range=0.05,
                             horizontal_flip=True,
                             fill_mode='nearest')

        #Train generator
        myGene = trainGenerator(1,
                                'data/train',
                                'image',
                                'label',
                                data_gen_args,
                                save_to_dir=None)

        #Check point
        now = datetime.now()
        cur_date = str(now.year) + '_' + str(now.month) + '_' + str(
            now.day) + '_' + str(now.hour) + '_' + str(now.minute)
        model_checkpoint = ModelCheckpoint('models/unet_stroke_' + cur_date +
                                           '_init.hdf5',
                                           monitor='loss',
                                           verbose=1,
                                           save_best_only=True)
        time_callback = TimeHistory()

        #Train model
        model.fit_generator(myGene,
                            steps_per_epoch=1,
                            epochs=10,
                            callbacks=[model_checkpoint, time_callback])

        #Data that we will send back
        data = [{
            'date': now,
            'model_name': 'unet_stroke_' + cur_date + '_init.hdf5',
            'from_scratch': True
        }]

        #Return result
        return jsonify(data)

#Get model training progress
@app.route('/training_progress', methods=['GET'])
def check_training_progress():

    response_object = {'current_epoch': TimeHistory.current_epoch}

    return jsonify(response_object)

if __name__ == '__main__':
    app.run(debug=True)