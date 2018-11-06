# based on: https://github.com/XD-DENG/flask-app-for-mxnet-img-classifier/blob/master/app.py

from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from collections import namedtuple
import mxnet as mx
import  hashlib
import datetime

from PIL import *

import matplotlib
matplotlib.use('Agg')

# fastai
from fastai import *
from fastai.vision import *
import torch
from pathlib import Path


# plotly plotting
import json
import plotly

import pandas as pd
import numpy as np
import plotly.graph_objs as go

app = Flask(__name__)
# restrict the size of the file uploaded
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024


################################################
# Error Handling
################################################

@app.errorhandler(404)
def FUN_404(error):
    return render_template("error.html")

@app.errorhandler(405)
def FUN_405(error):
    return render_template("error.html")

@app.errorhandler(500)
def FUN_500(error):
    return render_template("error.html")


################################################
# Functions for running classifier
################################################

# define a simple data batch
Batch = namedtuple('Batch', ['data'])

# # define the classes (TODO: read from file with model)
labels = ['fender_telecaster', 'gibson_les_paul', 'gibson_es', 
          'gibson_explorer', 'gibson_flying_v', 'fender_mustang', 
          'fender_stratocaster', 'gibson_sg', 'fender_jaguar', 
          'gibson_firebird', 'fender_jazzmaster']

# lookup
names = {'fender_telecaster': "Fender Stratocaster",
         'gibson_les_paul':   "Gibson Les Paul",
         'gibson_es':         "Gibson ES", 
         'gibson_explorer':   "Gibson Explorer",
         'gibson_flying_v':   "Gibson Flying V",
         'fender_mustang':    "Fender Mustang",
         'fender_stratocaster': 'Fender Stratocaster', 
         'gibson_sg':         "Gibson SG",
         'fender_jaguar':     "Fender Jaguar",
         'gibson_firebird':   "Gibson Firebird", 
         'fender_jazzmaster': "Gibson Jazzmaster"}

path = Path("/tmp")
data = ImageDataBunch.single_from_classes(path, labels, tfms=get_transforms(max_warp=0.0), size=299).normalize(imagenet_stats)
learner = create_cnn(data, models.resnet50)
learner.model.load_state_dict(
    torch.load("stage-3-50.pth", map_location="cpu")
)

# Prapare the MXNet model (pre-trained)
sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-152', 0)
mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], 
         label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)
with open('synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]


def get_image(file_location, local=False):
    # users can either 
    # [1] upload a picture (local = True)
    # or
    # [2] provide the image URL (local = False)
    if local == True:
        fname = file_location
    else:
        fname = mx.test_utils.download(file_location, dirname="static/img_pool")
    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)

    if img is None:
         return None
    
    # convert into format (batch, RGB, width, height)
    img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]

    return img


def mx_predict(file_location, local=False):
    img = get_image(file_location, local)

    # compute the predict probabilities
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()

    # Return the top-5
    prob = np.squeeze(prob)
    a = np.argsort(prob)[::-1]
    result = []
    for i in a[0:10]:
        result.append((labels[i].split(" ", 1)[1], round(prob[i], 3)))

    return result



def get_image_new(file_location, local=False):
    # users can either 
    # [1] upload a picture (local = True)
    # or
    # [2] provide the image URL (local = False)
    if local == True:
        fname = file_location
    else:
        fname = url_for(file_location, dirname="static", filename=img_pool + file_location)
    img = open_image(fname)
    
    if img is None:
         return None
    return img


def predict(file_location, local=False):
    img = get_image_new(file_location, local)

    pred_class, pred_idx, outputs = learner.predict(img)
    formatted_outputs = [x.numpy() * 100 for x in outputs] #torch.nn.functional.softmax(outputs, dim=0)]
    pred_probs = sorted(
            zip(learner.data.classes, formatted_outputs ),
            key=lambda p: p[1],
            reverse=True
        )

    formatted_outputs = [x.numpy() * 100 for x in torch.nn.functional.softmax(outputs, dim=0)]
    pred_probs2 = sorted(
            zip(learner.data.classes, formatted_outputs ),
            key=lambda p: p[1],
            reverse=True
    )


    return (pred_probs, names[pred_probs2[0][0]])

###### Plotting
def prediction_barchart(result):

    # data is list of name, value pairs
    y_values, x_values = map(list, zip(*result))
    # Create the Plotly Data Structure

    # classify based on prob.
    labels = ['Not sure', 'Well, maybe', 'Pretty sure', 'Trust me']
    cols   = ['red', 'orange', 'lightgreen', 'darkgreen']

    colors = dict(zip(labels, cols))
  
    
    bins = [0, 10, 25, 75, 100]

    # Build dataframe
    df = pd.DataFrame({'y': y_values,
                       'x': x_values,
                       'label': pd.cut(x_values, bins=bins, labels=labels)})

    bars = []
    for label, label_df in df.groupby('label'):
        bars.append(go.Bar(x=label_df.x[::-1],
                           y=label_df.y[::-1],
                           name=label,
                           marker={'color': colors[label]},
                           orientation='h'))

    graph = dict(
        data=bars,
        layout=dict(

            #title='Bar Plot',
            xaxis=dict(
                title="Probability"
            ),
            hovermode='y',
            showlegend=True,
            margin=go.Margin(
                l=150,
                r=10,
                t=10,
            )
        )
    )

    # Convert the figures to JSON
    return json.dumps(graph, cls=plotly.utils.PlotlyJSONEncoder)


################################################
# Functions for Image Archive
################################################

def FUN_resize_img(filename, resize_proportion = 0.3):
    '''
    FUN_resize_img() will resize the image passed to it as argument to be {resize_proportion} of the original size.
    '''
    img=cv2.imread(filename)
    small_img = cv2.resize(img, (0,0), fx=resize_proportion, fy=resize_proportion)
    cv2.imwrite(filename, small_img)

################################################
# Functions Building Endpoints
################################################

@app.route("/", methods = ['POST', "GET"])
def FUN_root():
	# Run correspoing code when the user provides the image url
	# If user chooses to upload an image instead, endpoint "/upload_image" will be invoked
    if request.method == "POST":
        img_url = request.form.get("img_url")
        #prediction_result = mx_predict(img_url)
        prediction_result, winner = predict(img_url)

        plotly_json = prediction_barchart(prediction_result)
        return render_template("index.html", img_src = img_url, 
                                             prediction_result = prediction_result,
                                             prediction_winner = prediction_winner,
                                             graphJSON=plotly_json)
    else:
        return render_template("index.html")


@app.route("/about/")
def FUN_about():
    return render_template("about.html")


ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'bmp']
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload_image", methods = ['POST'])
def FUN_upload_image():

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return(redirect(url_for("FUN_root")))
        file = request.files['file']

        # if user does not select file, browser also submit a empty part without filename
        if file.filename == '':
            return(redirect(url_for("FUN_root")))

        if file and allowed_file(file.filename):
            filename = os.path.join("static/img_pool", hashlib.sha256(str(datetime.datetime.now()).encode('utf-8')).hexdigest() + secure_filename(file.filename).lower())
            file.save(filename)
            #prediction_result = mx_predict(filename, local=True)

            prediction_result, prediction_winner = predict(filename, local=True)
            print(prediction_result)

            FUN_resize_img(filename)

            # create plotly chart
            plotly_json = prediction_barchart(prediction_result)
            print( prediction_result )
            print( plotly_json )
            return render_template("index.html", img_src = filename, 
                                                 prediction_result = prediction_result,
                                                 prediction_winner = prediction_winner,
                                                 graphJSON=plotly_json)
    return(redirect(url_for("FUN_root")))


################################################
# Start the service
################################################
if __name__ == "__main__":
    app.run(debug=True)