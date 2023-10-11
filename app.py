# pip install opencv-python opencv-contrib-python Pillow --user
# pip install flask flask-cors --user

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import json

from methods import search_similar_parts

app = Flask(__name__)
CORS(app)

@app.route('/process_image_matching', methods=['POST'])
def process_image_matching():
    try:
        # Get the uploaded blob image file
        image_file = request.files['image']
        # Get the selected region to match
        region = json.loads(request.form['region'])
        # Get the similarity metric
        sensitivity = float(request.form['sensitivity'])
        # Specify whether you will search with all rotations
        rotation = bool(request.form['rotation'])  # Convert to boolean
        # Specify whether you will search with grayscale image
        filter_color = bool(request.form['filter_color'])

        # Read the image using OpenCV
        image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

        unique_match_positions = search_similar_parts(image, region, sensitivity, rotation, filter_color)

        numpy_array = np.array(unique_match_positions, dtype=np.int64)

        # Prepare the results
        results = {
            'message': 'Image processing and similarity search completed successfully',
            'matches': json.dumps(numpy_array.tolist()),
        }
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
