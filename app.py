import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf
from skimage.io import imread
from skimage.transform import resize
import cv2
from PIL import Image
from io import BytesIO
import base64
from unet import get_unet
from tensorflow.keras.layers import Input

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model_cls = tf.keras.models.load_model('models/model_checkpoint.h5')
input_img = Input((256, 256, 3), name='img')
model_seg = get_unet(input_img, n_filters=16, dropout=0.2, batchnorm=True)
model_seg.load_weights('models/model-brain-mri.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(filepath)

        image = tf.keras.preprocessing.image.load_img(filepath, target_size=(224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        pred_cls = model_cls.predict(image)
        prediction = np.argmax(pred_cls)

        if prediction==0:
            result = 'glioma_tumor'
        elif prediction==1:
            result = 'meningioma_tumor'
        elif prediction==2:
            result = 'no_tumor'
        else:
            result = 'pituitary_tumor'

        image = cv2.imread(filepath)
        img = cv2.resize(image ,(256, 256))
        img = img / 255
        img = img[np.newaxis, :, :, :]
        pred_seg = model_seg.predict(img)

        # Reshape and convert data type for PIL image
        # pred_seg = pred_seg.reshape((256, 256)).astype(np.uint8)

        # result_image_pil = Image.fromarray(pred_seg)
        # result_image_pil = result_image_pil.resize((400, 400))
        pred_seg = (pred_seg * 255).astype(np.uint8)

        result_image_pil = Image.fromarray(pred_seg[0, :, :, 0], mode='L')  # Assuming the image is single-channel (grayscale)
        result_image_pil = result_image_pil.resize((350, 350))

        # Save the result image to a BytesIO object
        result_img_io = BytesIO()
        result_image_pil.save(result_img_io, 'JPEG')
        result_img_io.seek(0)

        # Encode the result image to base64 for embedding in HTML
        result_image_base64 = base64.b64encode(result_img_io.read()).decode('utf-8')

        return render_template('index.html', filename=filename, result=result, result_image=result_image_base64)

    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
