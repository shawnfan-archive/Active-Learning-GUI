#Imports
import requests
import json
import nibabel as nib
import numpy as np
from skimage.transform import resize

import skimage.io as io

#URL with the predict method
url = 'http://localhost:5000/getPrediction'

#Load in data (nifti)
img = nib.load('/Users/shawnfan/Dropbox/active_learning_20191115/Active-Learning-GUI/flask_unet/data/031923_t1w_deface_stx.nii.gz')
data = np.array(img.dataobj)
data = (data[:,:,120])
# print(len(data))
# print(len(data[0]))
# data dimensions: 197 by 233
data = resize(data, (256, 256), mode='reflect', preserve_range=True, order=3)
io.imsave(("output/input.png"), data/100)

# #Load in data (npy)
# data = np.load('data/test1_20191109.npy', allow_pickle=True)
# data = resize(data, (256, 256), mode='reflect', preserve_range=True, order=3)
# io.imsave(("output/input1_shawn.png"), data)

# #Ground truth
# img_truth = nib.load('data/031923_LesionSmooth_stx.nii.gz')
# data_truth = np.array(img_truth.dataobj)
# data_truth = (data_truth[:,:,120])
# data_truth = resize(data_truth, (256, 256), mode='reflect', preserve_range=True, order=3)
# io.imsave(("output/ground_truth.png"), data_truth/255)

#Jsonify
# j_data = json.dumps(str({'data': data}))
j_data = json.dumps(data.tolist())

#Create headers for json
headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}

#Send a post request
r = requests.post(url, data=j_data, headers=headers)

#Print statement
print(r, r.text)
