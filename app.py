from flask import Flask, jsonify, request
from flask_cors import CORS

Patients = {
        'subject_id': '1',
        'disease': "Parkinson's",
        'number_of_views': '100',
        'has_viewed': True,
        'data': {
            'x': [],
            'y': []
        }
}

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
        Patients['data']['x'] = post_data.get('x_coord')
        Patients['data']['y'] = post_data.get('y_coord')
        response_object['message'] = 'Progress saved!'
    else:
        response_object['patients'] = Patients
    return jsonify(response_object)

@app.route('/about_active_learning')
def about():
    return '<h1>About Acitve Learning</h1>'

if __name__ == '__main__':
    app.run(debug=True)