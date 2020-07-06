import os
from datetime import datetime
import numpy as np
import pandas as pd
from skimage.transform import resize
from unet.dice import *
import sqlite3
from sqlite_queries import *

#GUI canvas dimensions (These values must agree with front end)
canvas_width = 600
canvas_height = 500

def removeFile(filepath):
    """
    Check if a file exists and remove it if it exists

    Arguments: 
        filepath - file path of a file
    Returns: None
    """

    if os.path.exists(filepath):
        os.remove(filepath)

    # try:
    #     os.remove(filename)
    # except OSError as e: # this would be "except OSError, e:" before Python 2.6
    #     if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
    #         raise # re-raise exception if a different error occurred

def listDirectory(directory):
    """
    List all files in a directory and remove ".DS_Store"
    """

    items = os.listdir(directory)
    if ".DS_Store" in items:
        items.remove('.DS_Store')

    return items


def resizeArray(data, h, w):
    """
    Resize a binary prediction map as a NumPy array to h by w

    Arguments:
        data: image data as NumPy array
        h: height of resized image
        w: width of resized image

    Returns:
        rdata: resized image data as NumPy array
    """

    #! For binary maps, use an order of 0 - neartest neighbor and set anti_aliasing to False 
    #! Documentation: https://scikit-image.org/docs/dev/api/skimage.transform.html?ref=driverlayer.com/web#skimage.transform.resize
    rdata = resize(data, (h, w),
                   mode='reflect',
                   preserve_range=True,
                   order=0,
                   anti_aliasing=False)

    return rdata

def findLatestDir(search_dir):
    """
    """

    files = os.listdir(search_dir)

    #'.DS_Store' is part of Mac OS
    if '.DS_Store' in files:
        files.remove('.DS_Store')

    if len(files) == 0:
        return None

    files.sort(key=lambda file: datetime.strptime(file, '%Y_%m_%d_%H_%M'))

    return files[-1]

def findLatestOutput(search_dir = 'output'):
    """
    Find the latest output folder, return None if no output exists
    """

    files = os.listdir(search_dir)
    #'.DS_Store' file is part of Mac OS
    if '.DS_Store' in files:
        files.remove('.DS_Store')

    if len(files) == 0:
        return None

    files.sort(key=lambda file: datetime.strptime(file, '%Y_%m_%d_%H_%M'))

    return 'output/' + files[-1]


def loadActivationMap(filename):
    """
    Load activation map as NumPy array from file
    """
    # load activation map from numpy file
    numpy_activation = np.load(filename, allow_pickle=True)

    activation_converted = []
    for row in numpy_activation[0]:
        # numpy_activation is an array with 1 element
        row_converted = []
        for val in row:
            #Note: val is an array with 1 element
            if val[0] > 0.5:
                row_converted.append(1)
            else:
                row_converted.append(0)
        activation_converted.append(row_converted)

    return np.array(activation_converted)

# def updateLinkedLogs(db_path, log_names, column_names, entries):
#     """
#     Update an array of tables linked by FOREIGN KEY constraint in database
    
#     Arguments:
#         db_path:
#         log_names:
#         column_names:
#         entries:
    
#     Returns:
#         None
#     """

#     #Update first log
#     foreign_id = updateDB(db_path, log_names[0], column_names[0], entries[0])

#     linked_entry = []
#     for value in entries[1]:
#         linked_entry.append(value)
#     linked_entry.append(foreign_id)
#     linked_entry = tuple(linked_entry)

#     #Update second log
#     updateDB(db_path, log_names[1], column_names[1])

#     return None


db = "database/active_learning_20191210.db"
input_size = 192

def findLatestModel():
    """
    Find the file name of the latest model in /models

    Arguments: None
    Returns: 
        latest_model - the name of the latest model
    """

    model_paths = query_db(db, 'training_log', target_col='file_path', output_type=list)
    if len(model_paths) == 0:
        latest_model = "No model found. Please train a new model from scratch."
    else:
        latest_model = model_paths[-1].replace('models/', '')

    return latest_model

def loadLatestPrediction(img_id):
    """
    Load the latest prediction map as a NumPy array of an image with img_id
    """

    map_ids = query_db(db,
                       'image_to_map_log',
                       'map_id', ['image_id'], [img_id],
                       output_type=list)

    print(map_ids)

    map_paths_df = query_db(db, 'map_log', 'file_path, time_created',
                            ['map_id', 'is_manual'], [map_ids, 0])

    if len(map_paths_df) == 1:
        return np.zeros((input_size, input_size))

    #Sort by date and time created
    map_paths_df['time_created'] = pd.to_datetime(map_paths_df['time_created'],
                                                  format="%Y_%m_%d_%H_%M")
    prediction_path = map_paths_df.sort_values(
        'time_created', ascending=False).iloc[0]['file_path']
    print(f"Image ID: {img_id}. Latest Predition: {prediction_path}")

    prediction_np = loadActivationMap(prediction_path)

    return prediction_np


def loadLabel(img_id):
    """
    Load the label of an image with img_id

    Arguments: 
        img_id: an image ID
    
    Returns:
        label: the corresponding label map as a Numpy array
    """

    img_path = query_db(db,
                        'image_log',
                        'file_path', ['image_id'], [img_id],
                        output_type=str)

    label_path = img_path.replace('image', 'label')

    label = np.load(label_path)
    print(f"Image ID: {img_id}. Label: {label_path}")

    return label


def updateDiceScores(img_ids=[]):
    """
    Update DICE scores based on new predictions

    Arguments:
    Returns: 
    """
    print("Updating Dice scores...")

    if img_ids == []:
        img_ids = query_db(db, 'image_log', 'image_id', output_type=list)

    for img_id in img_ids:
        label = loadLabel(img_id)
        prediction = loadLatestPrediction(img_id)
        dice_score = dice(label, prediction)
        update_db(db, 'image_log', 'metric', dice_score, 'image_id', img_id)
        print(f"Image ID: {img_id}; Updated Dice Score: {dice_score}")

    print("Dice scores updated!")

    return None


def samplebyDice(sample_size):
    """
    Sample images with the lowest Dice score

    Arguments:
        sample_size: the number of images to sample
    Returns:
        images: 
    """
    # sorted_dice_score = {
    #     image_id: score
    #     for image_id, score in sorted(dice_score.items(),
    #                                   key=lambda item: item[1])
    # }
    # print(f"Sorted Dice Score: {sorted_dice_score}")
    # #Return image ids of images with 10 lowest DICE scores
    # image_ids = list(sorted_dice_score.keys())
    # image_ids = image_ids[:sample_size]

    # images = []
    # print("Sampled images:")

    img_df = query_db(db, "image_log")
    img_df.sort_values(by=['metric'], ascending=True)
    print(img_df)

    sampled_df = img_df.head(sample_size)

    images = []
    # for image_id in image_ids:
    for index, row in sampled_df.iterrows():
        file_path = row['file_path']

        print(file_path)
        # !- data sent to the front end
        filename = file_path.replace('data/train/image/', '')
        filename = filename.replace('.npy', '.png')

        #Resize image to canvas dimensions
        image_data = np.load(file_path)
        image_data = image_data * 80
        image_max = np.max(image_data)
        image_data = resizeArray(image_data, canvas_height, canvas_width)
        image_data = image_data.tolist()

        images.append({
            "image_id": row['image_id'],
            "disease": "Stroke",
            "path": filename,
            "data": image_data,
            "range": image_max + 1
        })

    return images

def sync_train_log():
    """
    Sync each table in database with the local directories

    Arguments: None
    Returns: None
    """

    print("Syncing traing log and train-to-image log with /models...")

    # try:
        #Find existing model paths
    model_names = listDirectory('models')
    model_paths_in_folder = []
    for model_name in model_names:
        model_paths_in_folder.append('models/' + model_name)
    model_paths_in_folder = set(model_paths_in_folder)
    print(f"Model paths in /models: {model_paths_in_folder}")

    con = sqlite3.connect(db)
    cur = con.cursor()

    training_df = query_db(db, 'training_log')
    model_paths_in_db = training_df['file_path'].tolist()
    model_paths_in_db = set(model_paths_in_db)
    print(f"Model paths in training log: {model_paths_in_db}")

    #Remove deleted models from training log and train_to_image log
    deleted_model_paths = model_paths_in_db.difference(model_paths_in_folder)
    print(f"Detected missing models: {deleted_model_paths}")
    for deleted_model_path in deleted_model_paths:
        
        training_id = training_df[training_df['file_path'] == deleted_model_path]['training_id'].iat[0] 

        sql_delete_query = "DELETE from train_to_image_log where training_id = " + str(training_id)
        cur.execute(sql_delete_query)

        sql_delete_query = "DELETE from training_log where training_id = " + str(training_id)
        cur.execute(sql_delete_query)

        print(f"Deleted fom training log: {deleted_model_path}")

    #Create entries for manually added models
    added_model_paths = model_paths_in_folder.difference(model_paths_in_db)
    print(f"Detected added models: {added_model_paths}")
    for added_model_path in added_model_paths:
        now = datetime.now()
        
        from_scratch = added_model_path.endswith('init.hdf5')
        print(from_scratch)

        training_entry = (str(now), added_model_path, from_scratch)
        insert_into_db(db, 'training_log', '(training_time, file_path, from_scratch)', training_entry)
        
        print(f"Added to training log: {added_model_path}")

    print("Finished syncing!")

    con.commit()
    con.close()
    # except sqlite3.Error as error:
    #     print('Failed to sync database', error)
    # finally:
    #     if (con):
    #         con.close()
    #         print(f"Finished syncing!")
    #     return "Training log successully synced!"

    return 'Training log successfully synced!'


def sync_map_log():
    """
    Sync map log and train/image (corrected maps) with train/label

    Arguments: None
    Returns: None
    """

    print(f"Syncing map log and image-to-map log with data/train...")

    #Map paths of maps in train/label
    map_names = listDirectory("data/train/label")
    map_paths_in_folder = []
    for map_name in map_names:
        map_paths_in_folder.append("data/train/label/" + map_name)
    map_paths_in_folder = set(map_paths_in_folder)

    #Map paths of corrected maps in database
    map_paths_in_db = query_db(db, 'map_log', 'file_path', ['is_manual'], [True], output_type = list)
    map_paths_in_db = set(map_paths_in_db)

    #Find paths of deleted paths
    deleted_paths = map_paths_in_db.difference(map_paths_in_folder)
    print(f"Detected missing maps:{deleted_paths}")

    for deleted_path in deleted_paths:
        #Delete from image folder
        deleted_path = deleted_path.replace('label', 'image')
        removeFile(deleted_path)
        print(f"Removed {deleted_path}")

    if len(deleted_paths) != 0:
        #Map IDs of deleted maps
        map_ids = query_db(db, 'map_log', 'map_id', ['file_path'], [list(deleted_paths)], output_type=list)
        
        #Remove entries from image_to_map log
        delete_from_db(db, 'image_to_map_log', 'map_id', map_ids)
        print("Updated image-to-map log!")

        #Remove entries from map_log
        delete_from_db(db, 'map_log', 'map_id', map_ids)
        print("Updated map log!")

    return "Image log successfully synced!"
