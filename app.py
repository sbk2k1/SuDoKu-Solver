import cv2
import numpy as np
from flask import Flask, redirect, render_template, Response, url_for, request, jsonify
from PIL import Image
import solve_sudoku_from_image
import base64
import io

app = Flask(__name__)
camera=cv2.VideoCapture(0)

def take_image():
    result, image = camera.read()
    if result:
        cv2.imwrite("SudokuImage.png", image)
    else:
        print("No image detected. Please! try again")

def generate_frames():
    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            ret,buffer=cv2.imencode('.png',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')

# index route
@app.route('/')
def index():
    return render_template('index.html')


@app.route("/solve", methods=["GET", "POST"])
def predict_img():
    message = request.get_json(force=True)
    encoded = message["image"]
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    #save image using opencv
    cv2.imwrite("SudokuImage.png", np.array(image))
    pred = solve_sudoku_from_image.solve_sudoku("SudokuImage.png")
    response = {
        'predictionImg': str(pred)
    }
    return jsonify(response)

# create a route to start webcam and take image
@app.route('/cam')
def cam():
    return render_template('cam.html')

@app.route("/click")
def click():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/image')
def image():
    take_image()
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)