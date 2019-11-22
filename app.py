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

# data = np.load("/Users/shawnfan/Downloads/test1.npy")
# io.imsave(("/Users/shawnfan/Downloads/postermap.png"), data)


output_path = "/Users/shawnfan/Dropbox/active_learning_20191115/Active-Learning-GUI/flask_unet/output"

# #activation map for brainpic0
# filename0 = "/Users/shawnfan/active_learning_prototype/prototype/src/assets/activation0_binary.txt"
# #activation map for brainpic1
# filename1 = "/Users/shawnfan/active_learning_prototype/prototype/src/assets/activation1_binary.txt"

#GUI canvas dimensions
canvas_width = 436
canvas_height = 364

image = {
    "image_id": "00001",
    'disease': "Stroke",
    'filename': ''
    # 'gz': ''.
    # 'jpeg': ''
}

activation_map = {
    'filename': 'flask_unet/output/resized_input.png',
    'activation': []
}

def resizeArray(data):
    data = resize(data, (canvas_height, canvas_width), mode='reflect', preserve_range=True, order=3)
    return data

def resizeImage(data):
    # resize brain image to GUI canvas size
    data = resize(data, (canvas_height, canvas_width), mode='reflect', preserve_range=True, order=3)
    io.imsave(("flask_unet/output/resized_input.png"), data/100)
    return None

# # Generate resized input.png
# img = nib.load('flask_unet/data/031923_t1w_deface_stx.nii.gz')
# data = np.array(img.dataobj)
# data = (data[:,:,120])
# resizeImage(data)

def extractActivationMap(filename):
    # load activation map from txt file
    activation_data = open(filename)
    lines = activation_data.readlines()
    activations = []

    for line in lines:
        line.replace('\n', '')
        split_line = line.split(',')
        row = []
        for element in split_line:
            row.append(float(element))
        activations.append(row)

    return activations

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
    files = os.listdir(output_path)
    files.remove('ex.ipynb')
    files.remove('ground_truth.png')
    files.remove('input.png')
    files.remove('resized_input.png')

    files.sort(key = lambda file: datetime.strptime(file, '%Y_%m_%d_%H_%M'))

    latest = files[-1]

    return latest

def getPrediction():
    #URL with the predict method
    url = 'http://localhost:5000/getPrediction'

    #Load in data (nifti)
    img = nib.load('flask_unet/data/031923_t1w_deface_stx.nii.gz')
    data = np.array(img.dataobj)
    data = (data[:,:,120])
    data = resize(data, (256, 256), mode='reflect', preserve_range=True, order=3)

    j_data = json.dumps(data.tolist())

    #Create headers for json
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}

    #Send a post request
    r = requests.post(url, data=j_data, headers=headers)    

    return None

def retrain():
    return None

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

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
        activation_map['corrected_activation'] = post_data.get(
            'corrected_activation')
        response_object['message'] = 'Progress saved!'
    else:
    # GET
        # find latest output folder and prediction.py
        latest_output = findLatestOutput()   
        activation_map['activation'] = loadActivationMap(output_path + '/' + latest_output + '/prediction.npy')
        # ActivationMaps['canvas_width'] = len(activation[0])
        # ActivationMaps['canvas_height'] = len(activation)
        response_object['activation_map'] = activation_map
        response_object['image'] = image
    return jsonify(response_object)

if __name__ == '__main__':
    app.run(debug=True)