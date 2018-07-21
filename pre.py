import numpy as np
import socket
from multiprocessing.dummy import Pool as ThreadPool
import sys
import urllib2
import cv2
import subprocess
 
invalid_path="vgg_face_dataset/invalid/"
folder_path="vgg_face_dataset/files/"

class Entry:
    def __init__(self,label,idnum,img,pose,detect_score,curation):
        self.label=label
        self.idnum=idnum
        self.img=img
        self.pose=pose
        self.detect_score=detect_score
        self.curation=curation

def get_filenames(batch):
    cmd = "ls "+folder_path+batch
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    filenames=output.split()
    print len(filenames)
    return filenames

def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    try:
        resp = urllib2.urlopen(url,timeout=10)
    except urllib2.HTTPError, e:
        print "error"
        return np.array([])
    except urllib2.URLError, e:
        print "error"
        return np.array([])
    except socket.timeout as e:
        print "error"
        return np.array([])
    except socket.error as e:
        print "error"
        return np.array([])
    except Exception:
        print "error"
        return np.array([])
    if resp is None:
        print "error"
        return np.array([])
    if resp.getcode()!=200:
        print "error"
        return np.array([])

    try:
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
    except urllib2.HTTPError, e:
        print "error"
        return np.array([])
    except urllib2.URLError, e:
        print "error"
        return np.array([])
    except socket.timeout as e:
        print "error"
        return np.array([])
    except socket.error as e:
        print "error"
        return np.array([])
    except Exception:
        print "error"
        return np.array([])
    if resp is None:
        print "error"
        return np.array([])
    if resp.getcode()!=200:
        print "error"
        return np.array([])

    if image.size == 0:
        print "error"
        return np.array([])
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if image is None:
        print "error"
        return np.array([])
    return image

def crop_and_scale(img,left,top,right,bottom):
    top=int(round(top))
    left=int(round(left))
    right=int(round(right))
    bottom=int(round(bottom))
    if img.shape[0]<bottom-top+1 or img.shape[1]<right-left+1 :
        print "error"
        return np.array([])
    crop_img = img[top:bottom+1, left:right+1]
    if crop_img.shape[0]<10 or crop_img.shape[1]<10:
        print "error"
        return np.array([])
    scale_img=cv2.resize(crop_img,(256,256))
    return scale_img

def process_files(files,batch):
    count = 0
    ivf = open(invalid_path+batch+".txt", 'w')
    data=list()
    for f in files:
        fcount = 0
        label = f[:-4]
        print label
        cmd = "cat "+folder_path+batch+"/"+f
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        entries = output.splitlines()
        print len(entries)
        for entry in entries:
            e=entry.split()
            idnum=int(e[0])
            print "url: "+e[1]
            l=float(e[2])
            t=float(e[3])
            r=float(e[4])
            b=float(e[5])
            if l<=0 or t<=0 or r<=0 or b<=0:
                print "error"
                cmd = "sed -i "+str(idnum-fcount)+"d "+folder_path+batch+"/"+f
                print cmd
                process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()
                ivf.write(label+" "+str(idnum)+"\n")
                fcount += 1
                continue
            raw_img=url_to_image(e[1])
            if raw_img.shape==(0,):
                cmd = "sed -i "+str(idnum-fcount)+"d "+folder_path+batch+"/"+f
                print cmd
                process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()
                ivf.write(label+" "+str(idnum)+"\n")
                fcount += 1
                continue
            else:
                img=crop_and_scale(raw_img,l,t,r,b)
                if img.shape==(0,):
                    cmd = "sed -i "+str(idnum-fcount)+"d "+folder_path+batch+"/"+f
                    print cmd
                    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
                    output, error = process.communicate()
                    ivf.write(label+" "+str(idnum)+"\n")
                    fcount += 1
                    continue
        count += fcount
    ivf.write("\ntotal: "+str(count)+"\n")
    ivf.close()
    print "finished getting invalid imgs"
    return data

def main():
    batch=sys.argv[1]
    files=get_filenames(batch)
    data=process_files(files,batch)
    print len(data)

main()
