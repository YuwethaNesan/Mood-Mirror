from flask import Flask, request, jsonify
from fer import FER
import numpy as np
import cv2
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

detector = FER()

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        image_data = data['image']
        image_data = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Image decoding failed"}), 400

        result = detector.detect_emotions(image)

        if result:
            emotions = result[0]["emotions"]
            return jsonify(emotions)
        else:
            return jsonify({"error": "No face detected"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
