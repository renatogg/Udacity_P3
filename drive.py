import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
import cv2
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
try:
    from msvcrt import getch  # try to import Windows version
except ImportError:
    def getch():   # define non-Windows version
        import tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
 
char = 0
# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf
import _thread
import time
from TrainingData import processImg
def keypress():
    global char
    while 1:
        char = getch()
        if char == b'\x03':
            break
    
 

_thread.start_new_thread(keypress, ())

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None
kp = 0.5
desired_speed = 10
steering_std = 0.5
@sio.on('telemetry')
def telemetry(sid, data):
    global desired_speed
    global steering_std
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = float(data["speed"])
    e = desired_speed - speed
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    h, w,l = image_array.shape
    img = cv2.resize(image_array,(int(w / 2),int(h / 2)), interpolation = cv2.INTER_AREA)
    h, w,l = img.shape
    img = img[int(h/2):,:,:]        
    img = processImg(img)
    
    
    
    #std = np.std(newimg)
    #newimg = newimg/std
    transformed_image_array = img[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))   

    #Proportional controller for throttle
    throttle = kp*e
    #DEBUG: force 12.5 degrees steering angle
    if char == b'a':
        steering_angle = -.5
    if char ==   b'd':
        steering_angle = .5
    #cut throttle for vehicle
    if char ==   b' ':
        throttle = 0
    #Increase / Decrease vehicle desired speed
    if char ==   b'w':
        desired_speed+=0.1
    if char ==   b's':
        desired_speed-=0.1
    print(char,steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
        #   model = model_from_json(json.loads(jfile.read()))\
        #
        # instead.
        model = model_from_json(jfile.read())


    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)