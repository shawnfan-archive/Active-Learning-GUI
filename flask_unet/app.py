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
import sqlite3

from tensorflow.python.keras.backend import set_session

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

#GUI canvas dimensions
canvas_width = 600
canvas_height = 500

images = []

for slice_id in range(100, 106):

    images.append({
        "image_id": str(slice_id - 99),
        "disease": "Stroke",
        "path": "Site6_031923___" + str(slice_id)
    })

activation_maps = {
    #image_id(string): activation map array
}


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


def loadActivationMap(filename):
    # load activation map from numpy file
    numpy_activation = np.load(filename)

    activation_converted = []
    for row in numpy_activation[0]:
        # numpy_activation is an array with 1 element
        row_converted = []
        for val in row:
            # val is an array with 1 element
            if val[0] > 0.5:
                row_converted.append(1)
            else:
                row_converted.append(0)
        activation_converted.append(row_converted)

    numpy_activation = np.array(activation_converted)
    resized_numpy_activation = resizeArray(numpy_activation)

    activation = resized_numpy_activation.tolist()

    return activation


def findLatestOutput():

    files = os.listdir('output')
    files.remove('.DS_Store')

    files.sort(key=lambda file: datetime.strptime(file, '%Y_%m_%d_%H_%M'))

    latest_output = files[-1]

    return latest_output


def findLatestModel():
    #Load in latest iteration of model
    df = pd.read_csv('models/model_tracking.csv', parse_dates=['date'])
    latest_model = df.sort_values('date',
                                  ascending=False).iloc[0]['model_name']

    return latest_model


def getPrediction():
    #URL with the predict method
    url = 'http://localhost:5000/getPrediction'

    post_data = {}

    # load nifti
    img = nib.load('data/031923_t1w_deface_stx.nii.gz')
    data = np.array(img.dataobj)

    for image_id in activation_maps:

        image_data = (data[:, :, int(image_id)+ 99])
        image_data = resize(image_data, (256, 256),
                            mode='reflect',
                            preserve_range=True,
                            order=3)

        post_data[image_id] = image_data.tolist()

    j_data = json.dumps(post_data)

    #Create headers for json
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}

    #Send a post request
    r = requests.post(url, data=j_data, headers=headers)

    return None


def updateTrainingData(corrected_activation_maps):

    conn = sqlite3.connect("database/active_learning_20191210.db")
    cur = conn.cursor()

    for image_id, activation_map in corrected_activation_maps.items():

        # convert activation_map(dict) to mapArray(list)
        mapArray = []

        for row_index in range(canvas_height):
            row = []
            for column_index in range(canvas_width):
                pixel_index = row_index * canvas_width + column_index
                row.append(activation_map[pixel_index])
            mapArray.append(row)

        mapNumPy = np.array(mapArray)
        mapNumPy = resizeArray(mapNumPy)

        # current time
        now = datetime.now()
        cur_date = str(now.year) + '_' + str(now.month) + '_' + str(
            now.day) + '_' + str(now.hour) + '_' + str(now.minute)

        # update map_log
        # file path for activation map
        map_path = 'data/train/label/' + cur_date + '_' + image_id + '.png'
        is_manual = True
        map_entry = (map_path, cur_date, is_manual)

        cur.execute(
            " INSERT INTO map_log (file_path, time_created, is_manual) VALUES"
            + str(map_entry))
        
        map_id = cur.lastrowid

        # save image of corrected activation map
        io.imsave(map_path, mapNumPy)

        image_to_map_entry = (image_id, map_id)

        # update image_to_map log
        cur.execute(" INSERT INTO image_to_map_log (image_id, map_id) VALUES" + str(image_to_map_entry))

        # load nifti
        img = nib.load('data/031923_t1w_deface_stx.nii.gz')
        data = np.array(img.dataobj)
        data = (data[:, :, int(image_id) + 99])
        data = resize(data, (256, 256),
                      mode='reflect',
                      preserve_range=True,
                      order=3)

        # file path for image
        image_path = 'data/train/image/' + cur_date + '_' + image_id + '.png'

        # save corresponding image to each corrected activation map
        io.imsave(image_path, data / 100)


    conn.commit()
    conn.close()

    return None


def retrain(from_scratch):

    graph = tf.get_default_graph()

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
    print(r.text)

    #Retrieve output and save/update model tracking csv
    decoded_model_info = json.loads(r.text)
    decoded_model_info = decoded_model_info[0]
    print(decoded_model_info)
    model_tracking = pd.read_csv('models/model_tracking.csv')
    model_tracking = model_tracking.append(json.loads(r.text))
    model_tracking.to_csv('models/model_tracking.csv', index=False)
    print('Model Data Saved!')

    #Update training log in database
    model_path = 'models/' + decoded_model_info['model_name']
    training_entry = (decoded_model_info['date'], model_path, from_scratch)

    conn = sqlite3.connect("database/active_learning_20191210.db")

    cur = conn.cursor()

    cur.execute(
        " INSERT INTO training_log (training_time, file_path, from_scratch) VALUES"
        + str(training_entry))

    training_id = cur.lastrowid

    #Update train-to-image log in database
    df = pd.read_sql_query(" SELECT * from image_log", conn)

    image_ids = df['image_id'].tolist()

    for image_id in image_ids:

        train_to_image_entry = (image_id, training_id)
        cur.execute(" INSERT INTO train_to_image_log (image_id, training_id) VALUES" + str(train_to_image_entry))

    conn.commit()
    conn.close()

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
#This is needed to ensure that you can retrain the model more than once
sess = tf.Session()


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

        corrected_activation_maps = post_data.get('activation_maps')
        from_scratch = post_data.get('from_scratch')

        updateTrainingData(corrected_activation_maps)

        retrain(from_scratch)

        getPrediction()

        response_object['message'] = 'Progress saved!'

    else:
        # GET
        response_object['images'] = images

        latest_model = findLatestModel()
        response_object['latest_model'] = latest_model

        latest_output = findLatestOutput()

        for image in images:
            activation_maps[image['image_id']] = loadActivationMap(
                'output/' + latest_output + '/' + image['image_id'] +
                '_prediction.npy')

        response_object['activation_maps'] = activation_maps

    return jsonify(response_object)


#API page
@app.route('/getPrediction', methods=['POST'])
#Prediction method
def makecalc():

    global graph
    global sess
    with graph.as_default():
        set_session(sess)

        #Current time
        now = datetime.now()
        cur_date = str(now.year) + '_' + str(now.month) + '_' + str(
            now.day) + '_' + str(now.hour) + '_' + str(now.minute)

        if not os.path.exists('output/' + cur_date):
            os.mkdir('output/' + cur_date)

        #Load in latest iteration of model
        latest_model = findLatestModel()

        model.load_weights('models/' + latest_model)

        #Retrieve request
        data = request.get_json()

        conn = sqlite3.connect("database/active_learning_20191210.db")
        cur = conn.cursor()

        for image_id in activation_maps:

            #Predict on data and save image
            pred = model.predict(
                np.array(data[image_id]).reshape(1, 256, 256, 1))
            io.imsave(
                ("output/" + cur_date + '/' + image_id + "_prediction.png"),
                pred[0, :, :, 0])

            #Return max value
            max_value = np.array2string(pred.max())

            #Save as numpy arrays
            np.save('output/' + cur_date + '/' + image_id + '_input.npy',
                    np.array(data))

            map_path = 'output/' + cur_date + '/' + image_id + '_prediction.npy'
            np.save(map_path, np.array(pred)) 

            is_manual = False

            map_entry = (map_path, cur_date, is_manual)

            cur.execute(" INSERT INTO map_log (file_path, time_created, is_manual) VALUES" + str(map_entry))

            map_id = cur.lastrowid

            image_to_map_entry = (image_id, map_id) 

            cur.execute(" INSERT INTO image_to_map_log (image_id, map_id) VALUES" + str(image_to_map_entry))                   

        conn.commit()
        conn.close()

        #Return result
        return 'New predictions uploaded!'


total_epochs = 1


#Callback that will keep track of training times
class TimeHistory(keras.callbacks.Callback):

    current_epoch = 0
    time_remaining = "Calculating..."

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

        TimeHistory.time_remaining = time.strftime(
            "%H:%M:%S",
            time.gmtime(elapsed * (total_epochs - TimeHistory.current_epoch)))

        TimeHistory.current_epoch = TimeHistory.current_epoch + 1

        #Write to file
        with open("testfile.txt", "a") as file:
            file.write("Elapsed Time: " + str(elapsed) + "\n" +
                       "Estimated Time Remaining:" +
                       TimeHistory.time_remaining + "\n")
            file.close()


#Retrain page
@app.route('/retrain', methods=['POST'])
#Prediction method
def retrain_model():

    global graph
    global sess
    with graph.as_default():
        set_session(sess)

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
                            epochs=total_epochs,
                            callbacks=[model_checkpoint, time_callback])

        #Data that we will send back
        data = [{
            'date': now,
            'model_name': 'unet_stroke_' + cur_date + '.hdf5',
            'from_scratch': False
        }]

        #Reinitialize current epoch and estimated time remaining
        TimeHistory.current_epoch = 0
        TimeHistory.time_remaining = "Calculating..."

        #Return result
        return jsonify(data)


#Train model from scratch
@app.route('/initialModel', methods=['POST'])
#Prediction method
def train_from_scratch():

    global graph
    global sess
    with graph.as_default():
        set_session(sess)

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
                            epochs=1,
                            callbacks=[model_checkpoint, time_callback])

        #Data that we will send back
        data = [{
            'date': now,
            'model_name': 'unet_stroke_' + cur_date + '_init.hdf5',
            'from_scratch': True
        }]

        #Reinitialize current epoch and estimated time remaining
        TimeHistory.current_epoch = 0
        TimeHistory.time_remaining = "Calculating..."

        #Return result
        return jsonify(data)


#Get model training progress
@app.route('/training_progress', methods=['GET'])
def check_training_progress():

    response_object = {
        'current_epoch': TimeHistory.current_epoch,
        'total_epochs': total_epochs,
        'time_remaining': TimeHistory.time_remaining
    }

    return jsonify(response_object)


if __name__ == '__main__':
    app.run(debug=True)