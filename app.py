from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np

ActivationMaps = {
        'image_id': '1',
        'disease': "Parkinson's",
        'number_of_views': '100',
        'has_viewed': True,
        'data': {},
        'activation': []
}

filename1 = "/Users/shawnfan/active_learning_prototype/prototype/src/assets/matrix1.txt"
brainMask = '/Users/shawnfan/active_learning_prototype/prototype/src/assets/brainMask.npy'

def extractPixels(markers):
    pixels =[]
    for marker in markers:
        radius = marker['markerSize']
        center_x = marker['x']
        center_y = marker['y']
        x_range = list(range(center_x - radius, center_x + radius + 1))
        y_range = list(range(center_y - radius, center_y + radius + 1))
        for x in x_range:
            for y in y_range:
                distance = (x - center_x)**2 + (y - center_y)**2
                if distance <= radius and [x, y] not in pixels:
                    pixels.append([x, y])
    return pixels

def computeActivation(filename):
    file = open(filename)
    lines = file.readlines()
    matrix = []
    for line in lines:
        processed_line = line.replace("\n", "")
        splitted_line = processed_line.split(',')
        row = []
        for number in splitted_line:
            row.append(float(number))
        matrix.append(row)

    # matrix = list(np.load(filename, allow_pickle=True))
    indices = []
    for row_number, row in enumerate(matrix):
        for column_number, element in enumerate(row):
            if element == 1:
                indices.append({'x': column_number, 'y': row_number, 'id': len(indices)})
        
    return indices

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
def all_patients():
    response_object = {'status': 'success'}
    if request.method == 'POST':
        post_data = request.get_json()
        ActivationMaps['data'] = extractPixels(post_data.get('corrections'))
        response_object['message'] = 'Progress saved!'
        response_object['activation_maps'] = ActivationMaps['data']
    else:
        ActivationMaps['activation'] = computeActivation(filename1)
        response_object['activation_maps'] = ActivationMaps
    return jsonify(response_object)

@app.route('/about_active_learning')
def about():
    return '<h1>About Acitve Learning</h1>'

if __name__ == '__main__':
    app.run(debug=True)