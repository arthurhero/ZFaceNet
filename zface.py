import requests
import cgi
from urllib.parse import quote_plus
from flask import Flask, request, redirect, url_for, render_template, Response, session
import json
import os
import os.path
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

import process_grinnell as pg

photo_path="http://184.105.225.21/grinnell_faces/processed/"
@zface.route('/zfacenet',methods=['POST'])
def zfacenet():
    pil_img = Image.open(request.files['file']).convert('RGB')
    cv_img = np.array(pil_img)
    cv_img = cv_img[:, :, ::-1]
    predicts = pg.predict(cv_img)
    urls=list()
    for p in predicts:
        urls.append(photo_path+p+".jpg")
    result = dict()
    result['names']=predicts
    result['urls']=urls
    return json.dumps(result)

if __name__ == "__main__":
    zface.run(debug==True)
