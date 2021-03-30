#NEW
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing.image import load_img, img_to_array
#from scipy.misc import imsave, imread, imresize
from skimage import transform, io
import numpy as np
import keras.models
import re
import base64

import sys 
import os
sys.path.append(os.path.abspath("./model"))
from load import *

app = Flask(__name__)
global model, graph
_, graph = init()
    
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict/', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
	#get data from drawing canvas and save as image
        parseImage(request.get_data())
	#read parsed image back in 8-bit, black and white mode (L)
        #x = io.imread('output.png', plugin= "pil")
        #x = np.invert(x)
        #x = load_img("output.png", grayscale=True, target_size=(28, 28))
        #x = transform.resize(x,(28,28))
    	# reshape image data for use in neural network
        from PIL import Image
        img_grey = Image.open('output.png').convert('L')
        img_grey = img_grey.resize((28, 28))
        x = img_to_array(img_grey)
        x = x.reshape(1,28,28,1)
        x = x.astype('float32')
        x= x / 255.0
        with graph.as_default(): #CHANGED graph.as_default to tf.Graph.as_de...
        
            #model.compile()
            #model.run_eagerly = True
            model, _ = init()
            out = model.predict_classes(x)
            response = str(out[0])
            #print(np.argmax(out, axis=1))
            #response = np.array_str(np.argmax(out, axis=1))
            #return JsonResponse({"output": response})
            print("This is the pred value: " + response) 
            return response 
        
def parseImage(imgData):
    # parse canvas bytes and save as output.png
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    with open('output.png','wb') as output:
        output.write(base64.decodebytes(imgstr))

if __name__ == '__main__':
    app.debug = False
    #app.run(debug=True, port=8000)
    print(("* Loading Keras model and Flask starting server...""please wait until server has fully started"))
    init()
    app.run(debug = True, port = 5000)
    #port = int(os.environ.get("PORT", 5000))
    #app.run(host='0.0.0.0', port=port)
