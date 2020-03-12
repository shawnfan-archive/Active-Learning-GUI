import os
from datetime import datetime
import numpy as np
import pandas as pd
from skimage.transform import resize
from unet.dice import *
import sqlite3


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
    rdata = resize(data, (h, w),
                   mode='reflect',
                   preserve_range=True,
                   order=0,
                   anti_aliasing=False)

    return rdata


def findLatestOutput():
    """
    Find the latest output folder, return None if no output exists
    """

    files = os.listdir('output')
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


def findLatestModel():
    """
    Load in latest iteration of model based on model_tracking.csv
    """
    df = pd.read_csv('models/model_tracking.csv', parse_dates=['date'])
    if df.shape[0] == 0:
        return "Latest model not found. Please train a new model from scratch."
    latest_model = df.sort_values('date',
                                  ascending=False).iloc[0]['model_name']

    return latest_model


def query_db(db_name,
             log_name,
             target_col='*',
             condition_col=None,
             condition_val=None,
             output_type=None):
    """
    """

    conn = sqlite3.connect(db_name)

    query = " SELECT " + target_col + " from " + log_name
    condition = ""
    if condition_col != None:
        condition = " where "
        for i in range(len(condition_col)):
            if i > 0:
                condition += " and "
            condition = condition + condition_col[i]
            if type(condition_val[i]) == list:
                condition = condition + " in " + str(tuple(condition_val[i]))
            else:
                condition = condition + " = " + str(condition_val[i])

    query = query + condition

    df = pd.read_sql_query(query, conn)

    conn.close()

    if output_type == str:
        output = df[target_col].to_string(index=False)
        output = output.replace(' ', '')  #Remove space
    elif output_type == list:
        output = df[target_col].to_list()
    else:
        output = df

    return output


def updateDB(db_path, log_name, col_names, entry):
    """
    Update a specific table in a database

    Arguments:
        db_path: database path 
        log_name: name of a table
        col_names: an array of names of all columns in the table
        entry: a tuple of values to be added to the table
    
    Returns:

    """

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(" INSERT INTO " + log_name + " " + str(col_names) +
                " VALUES " + str(entry))
    entry_id = cur.lastrowid

    conn.commit()
    conn.close()

    return entry_id


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


def loadLatestPrediction(img_id):
    """
    Load the latest prediction map as a NumPy array of an image with img_id
    """

    map_ids = query_db(db,
                       'image_to_map_log',
                       'map_id', ['image_id'], [img_id],
                       output_type=list)

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
    """
    print("Updating Dice Scores...")
    dice_score = {}

    if img_ids == []:
        img_ids = query_db(db, 'image_log', 'image_id', output_type=list)

    for img_id in img_ids:
        label = loadLabel(img_id)
        prediction = loadLatestPrediction(img_id)
        dice_score[img_id] = dice(label, prediction)

    print(f"Updated Dice Score: {dice_score}")

    return dice_score


def samplebyDice(dice_score, sample_size):
    """
    Sample images with the lowest Dice score
    """
    sorted_dice_score = {
        image_id: score
        for image_id, score in sorted(dice_score.items(),
                                      key=lambda item: item[1])
    }
    print(f"Sorted Dice Score: {sorted_dice_score}")
    #Return image ids of images with 10 lowest DICE scores
    image_ids = list(sorted_dice_score.keys())
    image_ids = image_ids[:sample_size]

    images = []
    print("Sampled images:")
    for image_id in image_ids:

        file_path = query_db(db,
                             'image_log',
                             'file_path', ['image_id'], [image_id],
                             output_type=str)
        print(file_path)
        filename = file_path.replace('data/train/image/', '')
        filename = filename.replace('.npy', '.png')

        images.append({
            "image_id": image_id,
            "disease": "Stroke",
            "path": filename
        })

    return images

def sync_db():
    """
    Sync each table in database with the local directories
    """

    try:
        con = sqlite3.connect(db)
        cur = con.cursor()
    except sqlite3.Error as error:
        print('Failed to sync database', error)
    finally:
        if (con):
            con.close()

    return None

    # def deleteRecord():
    # try:
    #     con = sqlite3.connect('active_learning_20191210.db')
    #     cursor = con.cursor()
    #     print("Connected to SQLite")

    #     # Deleting single record now
    #     sql_delete_query = """DELETE from training_log where training_id = 0"""
    #     cursor.execute(sql_delete_query)
    #     con.commit()
    #     print("Record deleted successfully ")
    #     cursor.close()

    # except sqlite3.Error as error:
    #     print("Failed to delete record from sqlite table", error)
    # finally:
    #     if (con):
    #         con.close()
    #         print("the sqlite connection is closed")