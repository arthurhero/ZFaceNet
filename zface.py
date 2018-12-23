import requests
import cgi
from flask import Flask, request, redirect, url_for, render_template, Response, session
import json
import os
from PIL import Image
import numpy as np

import process_grinnell as pg

photo_path="http://184.105.225.21/grinnell_faces/processed/"
zface=Flask(__name__)

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
    zface.debug=True
    zface.run(host='0.0.0.0',port=80)
