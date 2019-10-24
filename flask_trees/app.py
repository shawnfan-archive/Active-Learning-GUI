# Import libraries
import numpy as np
from flask import Flask, request, jsonify
import pickle

#Create flask
app = Flask(__name__)

#Load the model
model = pickle.load(open('models/final_prediction.pickle','rb'))

#API page
@app.route('/api',methods=['POST'])

#Prediction method
def makecalc():

    #Retrieve request
    data = request.get_json()

    #Predict and turn into string
    prediction = np.array2string(model.predict(data))

    #Return result
    return jsonify(prediction)
    
#Run app when called
if __name__ == '__main__':
    app.run(port=5000, debug=True)