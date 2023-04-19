import os
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


app = Flask(__name__)


model =load_model('braintumor3.h5',compile=False)
print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(class_idx):
    if class_idx == 0:
        return "glioma Tumor"
    elif class_idx == 1:
        return "Meningioma Tumor"
    elif class_idx==2:
        return "No tumor"
    else:
        return "Pitutary Tumor"


def getResult(img):
    # image=cv2.imread(img)
    # image = Image.fromarray(image, 'RGB')
    # image = image.resize((150, 150))
    # image=np.array(image)
    # input_img = np.expand_dims(image, axis=0)
    # result=model.predict(input_img)
    # return result
    image=cv2.imread(img)
    image=cv2.resize(image,(150,150))
    img_array=np.array(image)
    # img_array.shape
    img_array=img_array.reshape(1,150,150,3)
    # img_array.shape
    result=model.predict(img_array)
    class_idx = np.argmax(result) # get the index of the highest probability
    return class_idx
    


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value=getResult(file_path)
        class_name = get_className(value)
        return class_name
    return None


if __name__ == '__main__':
    app.run(debug=True)