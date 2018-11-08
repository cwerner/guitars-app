# based on: https://github.com/XD-DENG/flask-app-for-mxnet-img-classifier/blob/master/app.py

from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from collections import namedtuple
import  hashlib
import datetime

#from PIL import *
from PIL import Image as PILImage
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
#app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

################################################
# Error Handling
################################################

@app.errorhandler(404)
def FUN_404(error):
    return render_template("error.html"), 404

@app.errorhandler(405)
def FUN_405(error):
    return render_template("error.html"), 405

@app.errorhandler(413)
def FUN_413(error):
    return render_template("error.html"), 413

@app.errorhandler(500)
def FUN_500(error):
    return render_template("error.html"), 500


################################################
# Functions for running classifier
################################################

# define the classes (TODO: read from file with model)
labels = ['fender_telecaster', 'gibson_les_paul', 'gibson_es', 
          'gibson_explorer', 'gibson_flying_v', 'fender_mustang', 
          'fender_stratocaster', 'gibson_sg', 'fender_jaguar', 
          'gibson_firebird', 'fender_jazzmaster']

# lookup
names = {'fender_telecaster': "Fender Telecaster",
         'gibson_les_paul':   "Gibson Les Paul",
         'gibson_es':         "Gibson ES", 
         'gibson_explorer':   "Gibson Explorer",
         'gibson_flying_v':   "Gibson Flying V",
         'fender_mustang':    "Fender Mustang",
         'fender_stratocaster': 'Fender Stratocaster', 
         'gibson_sg':         "Gibson SG",
         'fender_jaguar':     "Fender Jaguar",
         'gibson_firebird':   "Gibson Firebird", 
         'fender_jazzmaster': "Fender Jazzmaster"}

MODEL = 'v0.1-stage-3-50.pth'

path = Path("/tmp")
data = ImageDataBunch.single_from_classes(path, labels, tfms=get_transforms(max_warp=0.0), size=299).normalize(imagenet_stats)
learner = create_cnn(data, models.resnet50)
learner.model.load_state_dict(
    torch.load("models/%s" % MODEL, map_location="cpu")
)

def get_image(file_location, local=False):
    # users can either 
    # [1] upload a picture (local = True)
    # or
    # [2] provide the image URL (local = False)
    if local == True:
        fname = file_location
    else:
        fname = url_for(file_location, dirname="static", filename=img_pool + file_location)
    
    if allowed_file(fname):
        img = open_image(fname)
    
        if img:
            return img
    return None


def predict(file_location, local=False, threshold=0.5):
    img = get_image(file_location, local)

    pred_class, pred_idx, outputs = learner.predict(img)

    #print('file_location:', file_location)
    #print('pred_class:', pred_class)
    #print('pred_idx:', pred_idx)
    #print('outputs:', outputs)

    formatted_outputs = [x.numpy() * 100 for x in torch.nn.functional.softmax(np.log(outputs), dim=0)]
    pred_probs = sorted(
            zip(learner.data.classes, formatted_outputs ),
            key=lambda p: p[1],
            reverse=True
        )

    winner_name = names[pred_probs[0][0]]
    if pred_probs[0][1] < threshold*100:
        winner_name = "Not Sure!" 

    return (pred_probs, winner_name)

###### Plotting
def prediction_barchart(result):
    #print(result)

    # data is list of name, value pairs
    y_values, x_values = map(list, zip(*result))
    # Create the Plotly Data Structure

    x_values = [x  if x < 0 else x for x in x_values]
    y_values = [names[y] for y in y_values]

    # classify based on prob.
    labels = ['Hm?', 'Maybe', 'Probably', 'Trust me']
    cols   = ['red', 'orange', 'lightgreen', 'darkgreen']

    colors = dict(zip(labels, cols))
  
    
    bins = [-0.001, 10, 25, 75, 100.001]

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
                title="Probability",
                range=[0, 100]
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

def FUN_resize_img(filename, resize_proportion = 0.5):
    '''
    FUN_resize_img() will resize the image passed to it as argument to be {resize_proportion} of the original size.
    '''
    im = PILImage.open(filename)
    basewidth = 300
    wpercent = (basewidth/float(im.size[0]))
    hsize = int((float(im.size[1])*float(wpercent)))
    im.thumbnail((basewidth,hsize), PILImage.ANTIALIAS)
    im.save(filename)


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
        prediction_result, prediction_winner = predict(img_url)

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
            FUN_resize_img(filename)

            # create plotly chart
            plotly_json = prediction_barchart(prediction_result)

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