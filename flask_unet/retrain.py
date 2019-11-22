#Imports
import requests
import json
import nibabel as nib
import numpy as np
from skimage.transform import resize

import pandas as pd
import skimage.io as io

#Whether training from scratch or retraining
from_scratch = False

if from_scratch:
    #URL with the predict method
    url = 'http://localhost:5000/initialModel'
else:
    url = 'http://localhost:5000/retrain'

#Load in data (nifti)
img = nib.load('data/031923_t1w_deface_stx.nii.gz')
data = np.array(img.dataobj)
data = (data[:,:,120])
data = resize(data, (256, 256), mode='reflect', preserve_range=True, order=3)
io.imsave(("output/input.png"), data/100)

#Jsonify
j_data = json.dumps(data.tolist())

#Create headers for json
headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}

#Send a post request
r = requests.post(url, data=j_data, headers=headers)

#Retrieve output and save/update model tracking csv
print(json.loads(r.text))
model_tracking = pd.read_csv('models/model_tracking.csv')
model_tracking = model_tracking.append(json.loads(r.text))
model_tracking.to_csv('models/model_tracking.csv', index=False)
print('Model Data Saved!')

