from flask import Flask, request
from PIL import Image
import detection

app = Flask(__name__, static_url_path='/public', static_folder='assets')


@app.route('/detect', methods=['POST'])
def detect():
    img = request.files.get('file')
    img = Image.open(img)
    result = detection.detect(img)

    return result


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6543)
