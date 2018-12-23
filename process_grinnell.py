import cv2
import numpy as np
import subprocess
import os
import sys

import res50 as r50

grinnell_path="grinnell_faces/unprocessed/"
grinnell_processed_path="grinnell_faces/processed/"
grinnell_score_file="grinnell_faces/score.txt"

def process_all():
    cmd = "ls "+grinnell_path
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    filenames=output.split()
    for f in filenames:
        img = cv2.imread(grinnell_path+f,cv2.IMREAD_UNCHANGED)
        img = img[65:218,27:180]
        img = cv2.resize(img,(224,224))
        cv2.imwrite(grinnell_processed_path+f,img)

def record_all_score():
    cmd = "ls "+grinnell_processed_path
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    filenames=output.split()
    with file(grinnell_score_file,'w') as outfile:
        for f in filenames:
            img = cv2.imread(grinnell_processed_path+f,cv2.IMREAD_UNCHANGED)
            img = img.transpose((2, 0, 1))
            score = r50.get_score(img)
            np.savetxt(outfile, score)

def predict(img):
    img = cv2.resize(img,(224,224))
    img = img.transpose((2, 0, 1))
    score = r50.get_score(img)
    cmd = "ls "+grinnell_path
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    filenames=output.split()
    cur_predict=""
    cur_dis_min=sys.maxint
    scores = np.loadtxt(grinnell_score_file).astype(np.float)
    for i in range(0,len(filenames)):
        s=scores[i]
        dis=(s-score).pow(2).sum()
        if dis<cur_dis_min:
            cur_dis_min=dis
            cur_predict=filenames[i][:-4]
    print "Prediction is "+cur_predict+"!!!"
    return cur_predict

#process_all()
#record_all_score()
