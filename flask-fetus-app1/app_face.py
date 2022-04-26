from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from PIL import Image

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = "I:\\guangzhoufuyou\\resize_all\\no_generator_segment\\modelsave\\area4\\resnet18\\ep010-val_acc0.899.h5"

# Load your trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
model = load_model(MODEL_PATH)
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = load_img(img_path, grayscale=False, color_mode='grayscale', target_size=None,interpolation='nearest')
    img = img.resize((220, 200), Image.ANTIALIAS)
    
    img= np.array(img)#将Image格式转换为ndarry格式
    img =img[:,:,np.newaxis] #[400,440]变为[400,440,1]
    img =img[np.newaxis,:,:]
    
    #归一化
    x = (img - np.min(img)) / (np.max(img) - np.min(img))
    # Preprocessing the image
    #x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    #x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        if preds[0][0] <= 0.7:    
        
           predict_label = "胎儿正常"
           pre_value = 0
           result = predict_label
        else:
           predict_label = "胎儿异常 参考风险值："
           pre_value = preds[0][0] 
           result = predict_label + str(pre_value)
        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class[0][0][1])               # Convert to string
        return result
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)
   #http://localhost:5000
    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
