from flask import Flask, jsonify, request
from flask_cors import CORS

#activation map for brainpic0
filename0 = "/Users/shawnfan/active_learning_prototype/prototype/src/assets/activation0_binary.txt"
#activation map for brainpic1
filename1 = "/Users/shawnfan/active_learning_prototype/prototype/src/assets/activation1_binary.txt"

ActivationMaps = {
    'image_id': '00001',
    'disease': "Parkinson's",
    'filename': filename1,
    'canvas_width': 0,
    'canvas_height': 0,
    'activation': [],
    'corrected_activation': []
}

def extractActivationMap(filename):
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
        ActivationMaps['corrected_activation'] = post_data.get(
            'corrected_activation')
        response_object['message'] = 'Progress saved!'
    else:
        activation = extractActivationMap(ActivationMaps['filename'])
        ActivationMaps['activation'] = activation 
        ActivationMaps['canvas_width'] = len(activation[0])
        ActivationMaps['canvas_height'] = len(activation)
        response_object['activation_map'] = ActivationMaps
    return jsonify(response_object)

@app.route('/about_active_learning')
def about():
    return '<h1>About Acitve Learning</h1>'

if __name__ == '__main__':
    app.run(debug=True)