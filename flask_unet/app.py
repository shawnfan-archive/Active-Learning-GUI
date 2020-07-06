from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from celery import Celery
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
from time_history import *

#TODO: Add DeepExplain to requirements.txt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

#model input dimensions
input_size = 192

#Database path
db = "database/active_learning_20191210.db"

#Number of images displayed on the GUI
#!- Do not set below 20 or risk Keras Progbar error
sample_size = 20

#Number of epochs for retraining (not from scratch)
nepochs_retrain = 10
#Number of epochs for initial training (from scratch)
nepochs_initial = 10

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
        # Loaded image already has the correct dimensions
        img_np = np.load(image_np_path, allow_pickle=True)
        # Convert to list for JSON
        post_data[image_id] = img_np.tolist()
        
    j_data = json.dumps(post_data)

    #Create headers for json
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}

    #Send a post request
    r = requests.post(prediction_url, data=j_data, headers=headers)

    decoded_info = json.loads(r.text)

    return decoded_info


def update_mnist_data():
    """
    """
    print("Saving")

    return None

def updateTrainingData(corrected_activation_maps):
    """
    Update training data and associated tables (map log and image-to-map log) in database
    """

    print("Updating training data...")

    print(corrected_activation_maps)

    for image_id, activation_map in corrected_activation_maps.items():

        now = datetime.now()
        cur_date = str(now.year) + '_' + str(now.month) + '_' + str(
            now.day) + '_' + str(now.hour) + '_' + str(now.minute)

        #Save corrected map as NumPy
        map_np = np.array(activation_map)
        map_np = resizeArray(map_np, input_size, input_size)
        map_path = 'data/train/label/' + cur_date + '_' + str(
                    image_id) + '.npy'
        # print(np.where(map_np != 0))
        np.save(map_path, map_np)
        print(f"Saving map of image {image_id} to {map_path}")

        #Update map_log
        is_manual = True
        map_id = insert_into_db(db, 'map_log',
                          '(file_path, time_created, is_manual)', (map_path, cur_date, is_manual))
        #Update image_to_map log
        insert_into_db(db, 'image_to_map_log', '(image_id, map_id)',
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

    Arguments:
    Returns:
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

    #Update training log in database
    model_path = 'models/' + decoded_model_info['model_name']
    training_entry = (decoded_model_info['date'], model_path, from_scratch)
    training_id = insert_into_db(db, 'training_log',
                           '(training_time, file_path, from_scratch)',
                           training_entry)

    #Update train-to-image log in database
    if not from_scratch:
        #Model retrained on corrected images only
        #activation_maps maps image ids to map arrays
        for img_id in activation_maps.keys():
            train_to_image_entry = (img_id, training_id)
            insert_into_db(db, 'train_to_image_log', '(image_id, training_id)',
                     train_to_image_entry)
    else:
        #Model trained on all images from scratch
        img_df = query_db(db, 'image_log')
        for index, row in img_df.iterrows():
            train_to_image_entry = (row['image_id'], training_id)
            insert_into_db(db, 'train_to_image_log', '(image_id, training_id)',
                     train_to_image_entry)

    return "Retrain request submitted!"

#Instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

#Enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

#Create model (imported from model.py)
model = Unet_origin()

#This is needed to ensure you are using the correct tensor graph
graph = tf.get_default_graph()
#This is needed to ensure that you can retrain the model more than once
sess = tf.Session()

@celery.task
def active_learning_post(corrected_activation_maps, from_scratch):
    """
    """

    print(corrected_activation_maps)

    image_ids = corrected_activation_maps.keys()
    image_ids = list(image_ids)

    #Update training data
    updateTrainingData(corrected_activation_maps)

    #Retrain model
    retrain(from_scratch, corrected_activation_maps, image_ids)

    #Make predictions
    getPrediction()

    #Update Dice scores
    updateDiceScores()

    return "Done!"


@app.route('/active_learning', methods=['GET', 'POST'])
def active_learning():

    response_object = {}
    if request.method == 'POST':
        # POST
        post_data = request.get_json()

        corrected_activation_maps = post_data.get('activation_maps')
        from_scratch = post_data.get('from_scratch')

        active_learning_post.delay(corrected_activation_maps, from_scratch)

        response_object['message'] = 'Training request accepted!'

    else:
        # GET
        sync_train_log()
        sync_map_log()

        #Sample images by DICE score
        images = samplebyDice(sample_size)
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
    """
    """

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

        prediction_progress = PredictionProgress(len(data))

        for image_id, image_data in data.items():

            pred = model.predict(
                np.array(image_data).reshape(1, input_size, input_size, 1))

            prediction_progress.on_prediction_end()

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

            map_id = insert_into_db(db, 'map_log',
                              '(file_path, time_created, is_manual)',
                              map_entry)

            #Update image_to_map log
            image_to_map_entry = (image_id, map_id)
            insert_into_db(db, 'image_to_map_log', '(image_id, map_id)',
                     image_to_map_entry)
        
        prediction_progress.on_completion()

        #! Flask function cannot end with returning None
        #TODO: Check if this fixes the json error
        return jsonify(map_ids)

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
        batch_size = 4
        train_params_multiclass = {
            'normalize': False,
            'batch_size': batch_size,
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
            patIDList.append(image_np_path)
        

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

        latest_model = findLatestModel()
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
            TimeHistory(nepochs_retrain, len(patIDList)/batch_size)
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
        batch_size = 4
        train_params_multiclass = {
            'normalize': False,
            'batch_size': batch_size,
            'n_classes': 1,
            'n_channels': 1,
            'shuffle': True
        }

        patIDList = os.listdir('data/train/image')
        patIDList = np.random.choice(patIDList, 24)  #choose 24 random subjects

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
            TimeHistory(nepochs_initial, len(patIDList)/batch_size)
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

        #Return result
        return jsonify(data)


@app.route('/training_progress', methods=['GET'])
def check_training_progress():
    """
    """
    try: 
        response_object = {
            'current_epoch': TimeHistory.current_epoch,
            'total_epochs': TimeHistory.total_epochs,
            'time_remaining': TimeHistory.time_remaining,
            'finished': TimeHistory.finished
        }
    except:
    #In case check_training_progress() is called before training begins
        response_object = {
            'currrent_epoch': 0,
            'total_epochs': 'x',
            'time_remaining': "Calculating...",
            'finished': False
        }
    
    print(response_object)

    return jsonify(response_object)


@app.route('/prediction_progress', methods=['GET'])
def check_prediction_progress():
    """
    """
    try: 
        response_object = {
            'progress': PredictionProgress.progress,
            'total': PredictionProgress.total,
            'finished': PredictionProgress.finished
        }
    except: 
        response_object = {
            'progress': 0,
            'total': 0,
            'finished': False
        }

    return jsonify(response_object)

@app.route('/playground', methods=["POST", "GET"])
def process_files_from_playground():
    """
    """

    if request.method == "POST":

        now = datetime.now()
        cur_date = str(now.year) + '_' + str(now.month) + '_' + str(
            now.day) + '_' + str(now.hour) + '_' + str(now.minute)
        new_dir = "data/playground/" + cur_date
        img_dir = new_dir + "/images/"
        map_dir = new_dir + "/maps/"
        os.makedirs(img_dir)
        os.makedirs(map_dir)
        
        print(f"New directory for images: {img_dir}")
        print(f"New directory for activation maps: {map_dir}")

        for i in range(len(request.files)):
            name = request.files[str(i)].filename
            if "img" in name: 
                request.files[str(i)].save(img_dir + name)
            else:
                request.files[str(i)].save(map_dir + name)

        return "File(s) received!"

    elif request.method == "GET":

        latest_dir = findLatestDir("data/playground")
        map_dir = "data/playground/" + latest_dir + "/maps"
        img_dir = "data/playground/" + latest_dir + "/images"

        response_object = {}

        images = []
        for img_name in listDirectory(directory=img_dir):
            img_id = img_name.split("_")[1].replace(".npy", "")
            images.append({
                "image_id": img_id, 
                "data": resizeArray(np.load(img_dir + "/" + img_name), h=500, w=500).tolist()
                })
        response_object["images"] = images
        response_object["height"] = len(images[0]["data"])
        response_object["width"] = len(images[0]["data"][0])

        activation_maps = {}
        for map_name in listDirectory(directory=map_dir):
            img_id = map_name.replace(".npy", "")
            activation_maps[img_id] = resizeArray(np.load(map_dir + "/" + map_name), h=500, w=500).tolist()
        response_object["activation_maps"] = activation_maps

        return jsonify(response_object)

@app.route('/playground_receiver', methods=["POST"])
def save_corrections_from_playground():

    post_data = request.get_json()

    corr_maps = post_data.get('activation_maps')

    latest_dir = findLatestDir("data/playground")
    corr_dir = "data/playground/" + latest_dir + "/corr"
    os.makedirs(corr_dir)

    for img_id, corr_map in corr_maps.items():
        np.save(corr_dir + "/" + str(img_id) + ".npy", np.array(corr_map))

    return "Corrected activation maps saved!"

if __name__ == '__main__':
    app.run(debug=True)