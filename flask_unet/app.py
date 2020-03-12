from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import pickle
import json
import nibabel as nib
import skimage.io as io
import tensorflow as tf
from unet.model import *
from unet.data import *
import keras
import time
import random
from tensorflow.python.keras.backend import set_session
from unet.DataGeneratorClass import *
from helpers import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

#GUI canvas dimensions (These values must agree with front end)
canvas_width = 600
canvas_height = 500

#model input dimensions
input_size = 192

#Database path
db = "database/active_learning_20191210.db"

#Number of images displayed on the GUI
sample_size = 20

#Number of epochs for retraining (not from scratch)
nepochs_retrain = 30
#Number of epochs for initial training (from scratch)
nepochs_initial = 30

def getPrediction():
    """
    Make predictions for all images
    """

    #URL with the predict method
    prediction_url = 'http://localhost:5000/getPrediction'

    img_df = query_db(db, 'image_log')

    post_data = {}
    #Make predictions on all images
    for index, row in img_df.iterrows():

        image_id = row['image_id']
        image_np_path = row['file_path']
        #Note: loaded image already has the correct dimensions
        img_np = np.load(image_np_path, allow_pickle=True)
        #Convert to list
        post_data[image_id] = img_np.tolist()

    # #Make predictions on samples only
    # for image in images:
    #     index = img_df[img_df['image_id'] == int(
    #         image['image_id'])].index.values.astype(int)[0]
    #     image_np_path = img_df.iloc[index]['file_path']
    #     img_np = np.load(image_np_path)
    #     post_data[image['image_id']] = img_np.tolist()

    j_data = json.dumps(post_data)

    #Create headers for json
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}

    #Send a post request
    r = requests.post(prediction_url, data=j_data, headers=headers)

    decoded_info = json.loads(r.text)

    return decoded_info['map_ids']


def updateTrainingData(corrected_activation_maps):
    """
    Update training data and associated tables (map log and image-to-map log) in database
    """

    print("Updating training data...")

    for image_id, activation_map in corrected_activation_maps.items():

        now = datetime.now()
        cur_date = str(now.year) + '_' + str(now.month) + '_' + str(
            now.day) + '_' + str(now.hour) + '_' + str(now.minute)

        #Save corrected map as NumPy
        map_np = np.array(activation_map)
        print(np.where(map_np != 0))
        map_np = resizeArray(map_np, input_size, input_size)
        map_path = 'data/train/label/' + cur_date + '_' + str(
                    image_id) + '.npy'
        print(np.where(map_np != 0))
        np.save(map_path, map_np)
        print(f"Saving map of image {image_id} to {map_path}")

        #Update map_log
        is_manual = True
        map_id = updateDB(db, 'map_log',
                          '(file_path, time_created, is_manual)', (map_path, cur_date, is_manual))
        #Update image_to_map log
        updateDB(db, 'image_to_map_log', '(image_id, map_id)',
                 (image_id, map_id))

        image_np_path = query_db(db, 'image_log', 'file_path', ['image_id'], [image_id], output_type = str)
        img_np = np.load(image_np_path)

        #File path for saving image
        image_path = 'data/train/image/' + cur_date + '_' + str(
            image_id) + '.npy'
        #Save image as NumPy
        np.save(image_path, img_np)
        print(f"Saving Image {image_id} to {image_path}")
        

    print("Training data updated!")
    return "Training data updated!"


def retrain(from_scratch, activation_maps, image_ids):
    """
    Send a POST request to the RETRAIN or TRAIN_FROM_SCRATCH URL
    """

    graph = tf.get_default_graph()

    if from_scratch:
        #URL with the predict method
        url = 'http://localhost:5000/initialModel'
        #Jsonify
        j_data = json.dumps('Initiate retraining model')
    else:
        url = 'http://localhost:5000/retrain'
        j_data = json.dumps(image_ids)

    #Create headers for json
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    #Send a post request
    r = requests.post(url, data=j_data, headers=headers)

    #Retrieve output and save/update model tracking csv
    decoded_model_info = json.loads(r.text)
    decoded_model_info = decoded_model_info[0]
    print(decoded_model_info)
    model_tracking = pd.read_csv('models/model_tracking.csv')
    model_tracking = model_tracking.append(json.loads(r.text))
    model_tracking.to_csv('models/model_tracking.csv', index=False)

    #Update training log in database
    model_path = 'models/' + decoded_model_info['model_name']
    training_entry = (decoded_model_info['date'], model_path, from_scratch)
    training_id = updateDB(db, 'training_log',
                           '(training_time, file_path, from_scratch)',
                           training_entry)

    #Update train-to-image log in database
    if not from_scratch:
        #Model retrained on corrected images only
        #activation_maps maps image ids to map arrays
        for img_id in activation_maps.keys():
            train_to_image_entry = (img_id, training_id)
            updateDB(db, 'train_to_image_log', '(image_id, training_id)',
                     train_to_image_entry)
    else:
        #Model trained on all images from scratch
        img_df = query_db(db, 'image_log')
        for index, row in img_df.iterrows():
            train_to_image_entry = (row['image_id'], training_id)
            updateDB(db, 'train_to_image_log', '(image_id, training_id)',
                     train_to_image_entry)

    return "Retrain request submitted!"

#Instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

#Enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

#Create model (imported from model.py)
model = Unet_origin()

#This is needed to ensure you are using the correct tensor graph
graph = tf.get_default_graph()
#This is needed to ensure that you can retrain the model more than once
sess = tf.Session()


#Sanity check
@app.route('/')
@app.route('/home')
def home():
    return 'Welcome to the Active Learning GUI!'


@app.route('/active_learning', methods=['GET', 'POST'])
def active_learning():

    response_object = {}
    if request.method == 'POST':
        # POST
        post_data = request.get_json()

        corrected_activation_maps = post_data.get('activation_maps')

        image_ids = corrected_activation_maps.keys()
        image_ids = list(image_ids)

        from_scratch = post_data.get('from_scratch')
        updateTrainingData(corrected_activation_maps)

        retrain(from_scratch, corrected_activation_maps, image_ids)

        getPrediction()

        response_object['message'] = 'Progress saved!'

    else:
        # GET
        #Sample images by DICE score
        dice_score = updateDiceScores()
        images = samplebyDice(dice_score, sample_size)
        response_object['images'] = images

        latest_model = findLatestModel()
        response_object['latest_model'] = latest_model

        #Send latest prediction maps or None if there is no prediction
        activation_maps = {}
        for image in images:
            latest_prediction = loadLatestPrediction(image['image_id'])
            latest_prediction = resizeArray(latest_prediction, canvas_height, canvas_width)
            activation_maps[image['image_id']] = latest_prediction.tolist()
        response_object['activation_maps'] = activation_maps

    return jsonify(response_object)


#API page
@app.route('/getPrediction', methods=['POST'])
#Prediction method
def makecalc():

    print("Making predictions...")

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

        map_ids = []

        for image_id, image_data in data.items():

            pred = model.predict(
                np.array(image_data).reshape(1, input_size, input_size, 1))
            io.imsave(("output/" + cur_date + '/' + str(image_id) +
                       "_prediction.png"), pred[0, :, :, 0])

            max_value = np.array2string(pred.max())

            #Save as numpy arrays
            #? Does this need to be saved? 
            np.save('output/' + cur_date + '/' + str(image_id) + '_input.npy',
                    np.array(image_data))

            #Update map log
            map_path = 'output/' + cur_date + '/' + str(
                image_id) + '_prediction.npy'
            np.save(map_path, np.array(pred))
            is_manual = False
            map_entry = (map_path, cur_date, is_manual)

            map_id = updateDB(db, 'map_log',
                              '(file_path, time_created, is_manual)',
                              map_entry)

            #Update image_to_map log
            image_to_map_entry = (image_id, map_id)
            updateDB(db, 'image_to_map_log', '(image_id, map_id)',
                     image_to_map_entry)

        return None


total_epochs = 1


#Callback that will keep track of training times
#Try to get time estimate before epoch starts
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


#Retrain only on the 10 corrected images
@app.route('/retrain', methods=['POST'])
def retrain_model():

    global graph
    global sess
    with graph.as_default():
        set_session(sess)

        #Input dimensions
        input_dim = (192, 192, 1)  #regularunet
        input_dim_for_data_gen = input_dim

        #Parameters
        train_params_multiclass = {
            'normalize': False,
            'batch_size': 4,
            'n_classes': 1,
            'n_channels': 1,
            'shuffle': True
        }

        image_ids = request.get_json()
        print(image_ids)

        patIDList = []
        for image_id in image_ids:
            image_np_path = query_db(db, 'image_log', 'file_path', ['image_id'], [image_id], output_type=str)
            image_np_path = image_np_path.replace('data/train/image/', '')
            print(image_np_path)
            patIDList.append(image_np_path)
        #patIDList = np.random.choice(patIDList, 24) #choose 24 random subjects

        #Generator
        train_generator = DataGenerator_stroke_unet(patIDList,
                                                    '',
                                                    data_dir='',
                                                    xls_filepath='',
                                                    dim=input_dim_for_data_gen,
                                                    **train_params_multiclass)

        #Model
        model = Unet_origin()
        print(model.summary())

        #Load in latest iteration of model
        df = pd.read_csv('models/model_tracking.csv', parse_dates=['date'])
        latest_model = df.sort_values('date',
                                      ascending=False).iloc[0]['model_name']
        model.load_weights('models/' + latest_model)
        model.compile(optimizer=Adam(lr=1e-5),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        #Check point
        now = datetime.now()
        cur_date = str(now.year) + '_' + str(now.month) + '_' + str(
            now.day) + '_' + str(now.hour) + '_' + str(now.minute)

        #Construct filename
        FILEPATH_MODEL = 'models/unet_stroke_' + cur_date + '.hdf5'

        #Callbacks
        callbacks_list = [
            ModelCheckpoint(FILEPATH_MODEL,
                            monitor='loss',
                            verbose=1,
                            save_best_only=True,
                            mode='auto'),
            TimeHistory()
        ]

        #################### RE-TRAIN ####################
        #Train
        model.fit_generator(generator=train_generator,
                            verbose=1,
                            epochs=nepochs_retrain,
                            callbacks=callbacks_list,
                            use_multiprocessing=False,
                            workers=4)

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


#Train model from scratch on all images (currently 24 random images)
@app.route('/initialModel', methods=['POST'])
def train_from_scratch():

    global graph
    global sess
    with graph.as_default():
        set_session(sess)

        #################### MODEL INPUTS ####################
        #Imput image
        input_dim = (192, 192, 1)  #regular unet
        # input_dim = (192, 192, 4) #dunet

        input_dim_for_data_gen = input_dim

        #Parameters
        train_params_multiclass = {
            'normalize': False,
            'batch_size': 4,
            'n_classes': 1,
            'n_channels': 1,
            'shuffle': True
        }

        patIDList = os.listdir('data/train/image')
        #patIDList = np.random.choice(patIDList, 24)  #choose 24 random subjects

        train_generator = DataGenerator_stroke_unet(patIDList,
                                                    '',
                                                    data_dir='',
                                                    xls_filepath='',
                                                    dim=input_dim_for_data_gen,
                                                    **train_params_multiclass)
        #Regular unet DataGenerator_stroke_unet
        #DUNET DataGenerator_stroke_d_unet

        #Model
        model = Unet_origin()
        model.compile(optimizer=Adam(lr=1e-5),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        print(model.summary())
        nepochs_initial = 10

        #Check point
        now = datetime.now()
        cur_date = str(now.year) + '_' + str(now.month) + '_' + str(
            now.day) + '_' + str(now.hour) + '_' + str(now.minute)
        FILEPATH_MODEL = 'models/unet_stroke_' + cur_date + '_init.hdf5'

        #Callbacks
        callbacks_list = [
            ModelCheckpoint(FILEPATH_MODEL,
                            monitor='loss',
                            verbose=1,
                            save_best_only=True,
                            mode='auto'),
            TimeHistory()
        ]

        #################### TRAIN ####################
        #Train
        model.fit_generator(generator=train_generator,
                            verbose=1,
                            epochs=nepochs_initial,
                            callbacks=callbacks_list,
                            use_multiprocessing=False,
                            workers=4)

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
    