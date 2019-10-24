#Imports
import requests
import json

#URL with the predict method
url = 'http://localhost:5000/api'

#Sample test data
data = [[14.34, 1.68, 2.7, 25.0, 98.0, 2.8, 1.31, 0.53, 2.7, 13.0, 0.57, 1.96, 660.0]]

#Jsonify
j_data = json.dumps(data)

#Create headers for json
headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}

#Send a post request
r = requests.post(url, data=j_data, headers=headers)

#Print statement
print(r, r.text)
