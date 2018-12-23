import cv2
import numpy as np
import subprocess
import os
import sys
import operator

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
    cur_predicts=list()
    scores = np.loadtxt(grinnell_score_file).astype(np.float)
    scores = scores.reshape((len(filenames),2622))
    for i in range(0,len(filenames)):
        s=scores[i]
        dis=(s-score).pow(2).sum()
        if i<5:
            cur_predicts.append((filenames[i][:-4],dis))
            cur_predicts=sorted(cur_predicts,key=operator.itemgetter(1))
        else:
            cur_predicts.append((filenames[i][:-4],dis))
            cur_predicts=sorted(cur_predicts,key=operator.itemgetter(1))
            cur_predicts=cur_predicts[0:5]
    #print "first prediction is "+cur_predicts[0][0]+"!!!"
    predicts=list()
    for p in cur_predicts:
        predicts.append(p[0])
    return predicts

#process_all()
#record_all_score()

'''
p=cv2.imread(grinnell_processed_path+"Jerod_Weinman.jpg",cv2.IMREAD_UNCHANGED)
predict(p)
'''
