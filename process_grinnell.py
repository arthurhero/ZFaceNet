import cv2
import numpy as np
import subprocess
import os
import sys

grinnell_path="grinnell_faces/unprocessed/"
grinnell_processed_path="grinnell_faces/processed/"

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

process_all()
