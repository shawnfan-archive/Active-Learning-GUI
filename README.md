# Active-Learning-GUI

### Flask Deep Learning Model

First install dependencies for the server
```python
conda create -n flask_unet python=3.7
conda activate flask_unet
pip install -r requirements.txt 
```

Then install the Redis message broker
```python
cd flask_unet
wget http://download.redis.io/releases/redis-5.0.8.tar.gz
tar xzf redis-5.0.8.tar.gz
cd redis-5.0.8
make
```

Run the following to start the server
```python
cd flask_unet
python app.py
```

Run the following to start the Redis message broker
```python 
cd redis-5.0.8
src/redis-server
```

Run the following to start the Celery worker
```python 
cd flask_unet
celery worker -A app.celery --loglevel=info
```

Lastly install dependencies for the app
```python
cd prototype
npm install --save vue-router
npm run serve
```

Note that you will need to install npm and conda to run the above. You can find find resources here: https://www.npmjs.com/get-npm
and https://www.anaconda.com/
