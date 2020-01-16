# Active-Learning-GUI

### Flask Deep Learning Model

First install dependencies for the server

```python
conda create -n flask_unet python=3.7
conda activate flask_unet
pip install -r requirements.txt 
pip install -U flask-cors
```

Run the following to start the server
```python
cd flask_unet
python app.py
```


Then install dependencies for the app
```python
cd prototype
npm install --save vue-router
npm run serve
```
