from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import tensorflow
import numpy as np

IMAGE_SIZE = (256, 256)

model = tensorflow.keras.models.load_model("model.h5")


def detect(image):
    img = prepare_image(image)
    predicts = model.predict(img)
    normalized = np.argmax(predicts, axis=1)

    return {
        'predict': normalized.tolist()[0],
    }


def prepare_image(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(IMAGE_SIZE)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = imagenet_utils.preprocess_input(img)

    return img
