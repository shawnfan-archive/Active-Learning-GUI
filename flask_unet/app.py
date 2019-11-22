# Import libraries
import numpy as np
from flask import Flask, request, jsonify
import pickle
from unet.model import *
from unet.data import *
import json
import tensorflow as tf
import pandas as pd

import skimage.io as io
from datetime import datetime

#Create flask
app = Flask(__name__)

#Create model
model = unet()

#This is needed to ensure you are using the correct tensor graph
graph = tf.get_default_graph()

#API page
@app.route('/getPrediction', methods=['POST'])

#Prediction method
def makecalc():

    global graph
    with graph.as_default():

        #Current time
        now = datetime.now()
        cur_date = str(now.year) + '_' + str(now.month) + '_' + str(now.day) + '_' + str(now.hour)+ '_' + str(now.minute)

        if not os.path.exists('output/' + cur_date):
            os.mkdir('output/' + cur_date)

        #Retrieve request
        data = request.get_json()

        #Load in latest iteration of model
        df = pd.read_csv('models/model_tracking.csv', parse_dates=['date'])
        latest_model = df.sort_values('date', ascending=False).iloc[0]['model_name']
        # latest_model = 'unet_stroke_20191108.hdf5'
        model.load_weights('models/' + latest_model)

        #Predict on data and save image
        pred = model.predict(np.array(data).reshape(1,256,256,1))
        io.imsave(("output/" + cur_date + '/' + "prediction.png"),pred[0,:,:,0])

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
        latest_model = df.sort_values('date', ascending=False).iloc[0]['model_name']
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
        myGene = trainGenerator(1,'data/train','image','label',data_gen_args,save_to_dir = None)

        #Check point
        now = datetime.now()
        cur_date = str(now.year) + '_' + str(now.month) + '_' + str(now.day) + '_' + str(now.hour)+ '_' + str(now.minute)
        model_checkpoint = ModelCheckpoint('models/unet_stroke_' + cur_date + '.hdf5', monitor='loss',verbose=1, save_best_only=True)
        
        #Train model
        model.fit_generator(myGene, steps_per_epoch=20, epochs=1, callbacks=[model_checkpoint])

        #Data that we will send back 
        data = [{'date': now, 'model_name': 'unet_stroke_' + cur_date + '.hdf5', 'from_scratch': False}]

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
        myGene = trainGenerator(1,'data/train','image','label',data_gen_args,save_to_dir = None)

        #Check point
        now = datetime.now()
        cur_date = str(now.year) + '_' + str(now.month) + '_' + str(now.day) + '_' + str(now.hour)+ '_' + str(now.minute)
        model_checkpoint = ModelCheckpoint('models/unet_stroke_' + cur_date + '_init.hdf5', monitor='loss',verbose=1, save_best_only=True)
        
        #Train model
        model.fit_generator(myGene, steps_per_epoch=200, epochs=1, callbacks=[model_checkpoint])

        #Data that we will send back 
        data = [{'date': now, 'model_name': 'unet_stroke_' + cur_date + '_init.hdf5', 'from_scratch': True}]

        #Return result
        return jsonify(data)


#Run app when called
if __name__ == '__main__':
    app.run(port=5000, debug=True)