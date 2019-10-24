from flask import Flask, jsonify, request
from flask_cors import CORS

ActivationMaps = {
        'image_id': '1',
        'disease': "Parkinson's",
        'number_of_views': '100',
        'has_viewed': True,
        'data': {}
}

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
    else:
        response_object['activation_maps'] = ActivationMaps
    return jsonify(response_object)

@app.route('/about_active_learning')
def about():
    return '<h1>About Acitve Learning</h1>'

if __name__ == '__main__':
    app.run(debug=True)