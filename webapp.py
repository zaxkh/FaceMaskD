from flask import Flask, render_template, request, send_from_directory, jsonify
import cv2
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten
import numpy as np
import os

# Setup of nerural network layers
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64,64,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load trained model
model.load_weights('static/modelBest.h5')


COUNT = 0

#Configuration handle
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

def face_detection(image_file):
    # Simple helper function To detect Faces
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    img = cv2.imread(image_file)

    # Detect faces
    faces = face_cascade.detectMultiScale(img, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    path = os.path.join(os.getcwd(), "static")
    cv2.imwrite(os.path.join(path, "detected.png"), img)
    return img

# Servers Homepage
@app.route('/')
def man():
    return render_template('index.html')


@app.route('/home', methods=['POST', "GET"])
def home():
    global COUNT
    img = request.files['image']

    # Check what is response Json or Template
    resp_type = request.form.get("resp-type") 

    # If response type is Json it will send JSON response, if its the template IT will render a template.
    # Saves uploaded image
    img.save('static/{}.jpg'.format(COUNT)) 

    face_detection('static/{}.jpg'.format(COUNT))
    detected = os.getcwd() + "/detected.png"

    #Resizes and shape image   
    img_arr = cv2.imread('static/{}.jpg'.format(COUNT))

    img_arr = cv2.resize(img_arr, (64,64))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 64,64,3)

    # Validate Image with trained model loaded
    prediction = model.predict(img_arr)

    # Save results and display the rounded values of the probability of the prediction
    test = np.array([round(prediction[0,0], 2),round(prediction[0,1], 2)])
    COUNT += 1

    # Responds with json
    if resp_type == "json": 
        data = {
          "mask Worn": str(test[0]) + "%"  
        }
        return jsonify(data)
    return render_template('prediction.html', data=test)


@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static', "{}.jpg".format(COUNT-1))


if __name__ == '__main__':
    app.run(debug=True)



