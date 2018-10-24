import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import timedelta
import math
import socket
from multiprocessing import Pool
import sys
import urllib2
import cv2
import subprocess
import random

avg_path="vgg_face_dataset/avg/"
folder_path="vgg_face_dataset/files_10/"
validation_path="vgg_face_dataset/validation_10/"
test_path="vgg_face_dataset/test_10/"

#mini_batch_size  = 64
mini_batch_size  = 32

orig_img_size=256
img_size=224
num_channels = 3
#num_classes = 2622
num_classes = 10

#data augmentation
flip_chance = 0.5 

def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    try:
        resp = urllib2.urlopen(url,timeout=10)
    except urllib2.HTTPError, e:
        return np.array([])
    except urllib2.URLError, e:
        return np.array([])
    except socket.timeout as e:
        return np.array([])
    except socket.error as e:
        return np.array([])
    except Exception:
        return np.array([])
    if resp is None:
        return np.array([])
    if resp.getcode()!=200:
        return np.array([])

    try:
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
    except urllib2.HTTPError, e:
        return np.array([])
    except urllib2.URLError, e:
        return np.array([])
    except socket.timeout as e:
        return np.array([])
    except socket.error as e:
        return np.array([])
    except Exception:
        return np.array([])
    if resp is None:
        return np.array([])
    if resp.getcode()!=200:
        return np.array([])

    if image.size == 0:
        return np.array([])
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if image is None:
        return np.array([])
    return image

def crop_and_scale(img,left,top,right,bottom):
    top=int(round(top))
    left=int(round(left))
    right=int(round(right))
    bottom=int(round(bottom))
    if img.shape[0]<bottom-top+1 or img.shape[1]<right-left+1 :
        return np.array([])
    crop_img = img[top:bottom+1, left:right+1]
    if crop_img.shape[0]<10 or crop_img.shape[1]<10:
        return np.array([])
    scale_img=cv2.resize(crop_img,(orig_img_size,orig_img_size))
    real_avg=cv2.imread(avg_path+"real_avg.png",cv2.IMREAD_UNCHANGED)
    scale_img=np.int_(scale_img)
    real_avg =np.int_(real_avg)
    result_img = scale_img-real_avg
    result_img = result_img/255.0 
    return result_img

def under_prob(prob):
    x=random.randint(0,9999)
    return x<prob*10000

def random_proc(img):
    x_start=random.randint(0,orig_img_size-img_size)
    y_start=random.randint(0,orig_img_size-img_size)
    crop_img = img[x_start:x_start+img_size, y_start:y_start+img_size]
    if under_prob(flip_chance):
        flip_img = cv2.flip(crop_img, 1)
        return flip_img
    return crop_img

def get_one_img_by_name(person):
    while True:
        cmd = "cat "+folder_path+person+".txt"
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        entries = output.splitlines()
        entry_total = len(entries)
        entry_num =  random.randint(0,entry_total-1)
        entry = entries[entry_num]
        e=entry.split()
        l=float(e[2])
        t=float(e[3])
        r=float(e[4])
        b=float(e[5])
        if l<=0 or t<=0 or r<=0 or b<=0:
            continue
        raw_img=url_to_image(e[1])
        if raw_img.shape==(0,):
            continue
        else:
            img=crop_and_scale(raw_img,l,t,r,b)
            if img.shape==(0,):
                continue
            else:
                img=random_proc(img)
                img = img.transpose((2, 0, 1))
                return img

def get_one_sample(filenames):
    person_num = random.randint(0,num_classes-1)
    person = filenames[person_num][:-4]
    img = get_one_img_by_name(person)
    return img,person_num

def get_mini_batch():
    start=time.time()
    imgs = list()
    labels = list()
    cmd = "ls "+folder_path
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    filenames=output.split()
    pool = Pool(processes=5)
    results = pool.map(get_one_sample, [filenames]*mini_batch_size)
    for pair in results:
        img, label = pair
        imgs.append(img)
        labels.append(label)
    end=time.time()
    #print 'got batch time: '+str(end-start)
    #print imgs[0]
    return imgs, labels

def get_one_test_sample(filenames,path):
    while True:
        file_num=random.randint(0,len(filenames)-1)
        filename=filenames[file_num]
        cmd = "cat "+path+"/"+filename
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        entries = output.splitlines()
        entry_total = len(entries)
        entry_num =  random.randint(0,entry_total-1)
        entry = entries[entry_num]
        e=entry.split()
        person_num=int(e[0])
        l=float(e[4])
        t=float(e[5])
        r=float(e[6])
        b=float(e[7])
        if l<=0 or t<=0 or r<=0 or b<=0:
            continue
        raw_img=url_to_image(e[3])
        if raw_img.shape==(0,):
            continue
        else:
            img=crop_and_scale(raw_img,l,t,r,b)
            if img.shape==(0,):
                continue
            else:
                img=random_proc(img)
                img = img.transpose((2, 0, 1))
                return img,person_num

def get_one_test_sample_wrapper(args):
    return get_one_test_sample(*args)

def get_test_batch(vali=False):
    start=time.time()
    imgs = list()
    labels = list()
    path=""
    if (vali):
        path = validation_path
    else:
        path = test_path
    cmd = "ls "+path
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    filenames=output.split()
    pool = Pool(processes=5)
    results = pool.map(get_one_test_sample_wrapper, [(filenames,path)]*mini_batch_size)
    for pair in results:
        img, label = pair
        imgs.append(img)
        labels.append(label)
    end=time.time()
    #print 'got test batch time: '+str(end-start)
    return imgs, labels

def num_to_name(num):
    cmd = "ls "+folder_path
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    filenames=output.split()
    return filenames[num][:-4]

######################Triplet loss
def get_one_triplet(filenames):
    person_num_p = random.randint(0,num_classes-1)
    person_num_n = 0
    while True:
        person_num_n = random.randint(0,num_classes-1)
        if person_num_p != person_num_n:
            break
    person_p = filenames[person_num_p][:-4]
    person_n = filenames[person_num_n][:-4]
    img1 = get_one_img_by_name(person_p)
    img2 = get_one_img_by_name(person_p)
    img3 = get_one_img_by_name(person_n)
    return img1,img2,img3

    '''
def get_triplet_batch():
    start=time.time()
    cmd = "ls "+folder_path
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    filenames=output.split()
    person_num_p = random.randint(0,num_classes-1)
    person_num_n = 0
    while True:
        person_num_n = random.randint(0,num_classes-1)
        if person_num_p != person_num_n:
            break
    person_p = filenames[person_num_p][:-4]
    person_n = filenames[person_num_n][:-4]
    pool = Pool(processes=5)
    img1s = pool.map(get_one_img_by_name, [person_p]*mini_batch_size)
    img2s = pool.map(get_one_img_by_name, [person_p]*mini_batch_size)
    img3s = pool.map(get_one_img_by_name, [person_n]*mini_batch_size)
    end=time.time()
    #print 'got batch time: '+str(end-start)
    return img1s,img2s,img3s 

'''
def get_triplet_batch():
    start=time.time()
    img1s=list()
    img2s=list()
    img3s=list()
    cmd = "ls "+folder_path
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    filenames=output.split()
    pool = Pool(processes=5)
    results = pool.map(get_one_triplet, [filenames]*mini_batch_size)
    for img1,img2,img3 in results:
        img1s.append(img1)
        img2s.append(img2)
        img3s.append(img3)
    end=time.time()
    #print 'got batch time: '+str(end-start)
    return img1s,img2s,img3s 

def get_one_pair(p1,p2):
    img1 = get_one_img_by_name(p1)
    img2 = get_one_img_by_name(p2)
    return img1,img2
