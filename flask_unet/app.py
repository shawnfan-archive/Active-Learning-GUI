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

#GUI canvas dimensions
canvas_width = 436
canvas_height = 364

images = []
for image_id in range(100, 106):
    images.append({
        "image_id": str(image_id),
        "disease": "Stroke",
        "path": "Site6_031923___" + str(image_id)
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
    files.remove('ex.ipynb')
    files.remove('ground_truth.png')
    files.remove('input.png')
    files.remove('resized_input.png')
    files.remove('.DS_Store')

    print(files)

    files.sort(key=lambda file: datetime.strptime(file, '%Y_%m_%d_%H_%M'))

    latest_output = files[-1]

    return latest_output


def findLatestModel():
    #Load in latest iteration of model
    df = pd.read_csv('models/model_tracking.csv', parse_dates=['date'])
    latest_model = df.sort_values('date',
                                  ascending=False).iloc[0]['model_name']

    return latest_model


# load nifti
img = nib.load('data/031923_t1w_deface_stx.nii.gz')
data = np.array(img.dataobj)
image_data = (data[:, :, 106])

io.imsave(("data/new_image.png"), image_data)


def getPrediction():
    #URL with the predict method
    url = 'http://localhost:5000/getPrediction'

    post_data = {}

    # load nifti
    img = nib.load('data/031923_t1w_deface_stx.nii.gz')
    data = np.array(img.dataobj)

    for image_id in activation_maps:

        image_data = (data[:, :, int(image_id)])
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

        # save corrected activation map
        io.imsave('data/train/label/' + cur_date + '_' + image_id + '.jpeg',
                  mapNumPy)

        # load nifti
        img = nib.load('data/031923_t1w_deface_stx.nii.gz')
        data = np.array(img.dataobj)
        data = (data[:, :, int(image_id)])
        data = resize(data, (256, 256),
                      mode='reflect',
                      preserve_range=True,
                      order=3)
        # save corresponding image to each corrected activation map
        io.imsave('data/train/image/' + cur_date + '_' + image_id + '.jpeg',
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

        corrected_activation_maps = post_data.get('activation_maps')
        from_scratch = post_data.get('from_scratch')

        updateTrainingData(corrected_activation_maps)

        retrain(from_scratch)

        # # add unseen images
        # images.append({
        #     "image_id": str(106),
        #     "disease": "Stroke",
        #     "path": "Site6_031923___" + str(106)
        # })

        # images.append({
        #     "image_id": str(120),
        #     "disease": "Stroke",
        #     "path": "Site6_031923___" + str(120)
        # })

        # activation_maps["106"] = None
        # activation_maps["120"] = None

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
    with graph.as_default():

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
            np.save('output/' + cur_date + '/' + image_id + '_prediction.npy',
                    np.array(pred))

        #Return result
        return 'New predictions uploaded!'


total_epochs = 10


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
                            epochs=total_epochs,
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

    response_object = {
        'current_epoch': TimeHistory.current_epoch,
        'total_epochs': total_epochs,
        'time_remaining': TimeHistory.time_remaining
    }

    return jsonify(response_object)


if __name__ == '__main__':
    app.run(debug=True)