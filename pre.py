import numpy as np
import urllib2
import cv2
import subprocess

invalid_path="vgg_face_dataset/invalid.txt"
folder_path="vgg_face_dataset/files/test/"

SIZE=256

class Entry:
    def __init__(self,label,idnum,img,pose,detect_score,curation):
        self.label=label
        self.idnum=idnum
        self.img=img
        self.pose=pose
        self.detect_score=detect_score
        self.curation=curation

def get_filenames():
    cmd = "ls "+folder_path
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
    except Exception:
        print "error"
        return np.array([])
    if resp is None:
        print "error"
        return np.array([])
    if resp.getcode()!=200:
        print "error"
        return np.array([])
    '''
    if resp.read() is None:
        print "error"
        return np.array([])
    '''
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
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
    scale_img=cv2.resize(crop_img,(SIZE,SIZE))
    return scale_img

def get_invalid(files):
    ivf = open(invalid_path, 'w')
    for f in files:
        label = f[:-4]
        print label
        cmd = "cat "+folder_path+f
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
                ivf.write(label+" "+str(idnum)+"\n")
                continue
            raw_img=url_to_image(e[1])
            if raw_img.shape==(0,):
                ivf.write(label+" "+str(idnum)+"\n")
                continue
            else:
                img=crop_and_scale(raw_img,l,t,r,b)
                if img.shape==(0,):
                    ivf.write(label+" "+str(idnum)+"\n")
                    continue
    ivf.close()
    print "finished getting invalid imgs"

def calculate_avg(files):
    cmd = "cat "+invalid_path
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    invalids = output.splitlines()
    N = 20*len(files)-len(invalids)
    avg_img=np.zeros((SIZE,SIZE,3),np.float)
    print str(N)+" valid pics"
    invalid_map=dict()
    for invalid in invalids:
        pair = invalid.split()
        key = pair[0]
        val = int(pair[1])
        if key in invalid_map:
            vals = invalid_map[key]
            vals.append(val)
            invalid_map[key]=vals
        else:
            vals = list()
            vals.append(val)
            invalid_map[key]=vals
    for f in files:
        label = f[:-4]
        print label
        cmd = "cat "+folder_path+f
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        entries = output.splitlines()
        for entry in entries:
            e=entry.split()
            idnum=int(e[0])
            if label in invalid_map and idnum in invalid_map[label]:
                continue
            l=float(e[2])
            t=float(e[3])
            r=float(e[4])
            b=float(e[5])
            raw_img=url_to_image(e[1])
            if raw_img.shape==(0,):
                continue
            else:
                img=crop_and_scale(raw_img,l,t,r,b)
                if img.shape==(0,):
                    continue
                float_img=img.astype(np.float)
                avg_img+=float_img/N
    avg_img=np.array(np.round(avg_img),dtype=np.uint8)
    return avg_img

def main():
    files=get_filenames()
    #get_invalid(files)
    avg_img=calculate_avg(files)
    print "got avg!"
    cv2.imwrite('avg_img.png',avg_img)
    cv2.imshow('window',avg_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()
