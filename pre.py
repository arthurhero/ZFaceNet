import numpy as np
import socket
from multiprocessing.dummy import Pool as ThreadPool
import sys
import urllib2
import cv2
import subprocess
import random
 
invalid_path="vgg_face_dataset/invalid/"
avg_path="vgg_face_dataset/avg/"
folder_path="vgg_face_dataset/files/"
validation_path="vgg_face_dataset/validation/"
test_path="vgg_face_dataset/test/"

NUM_PER_FILE=1000
SIZE=256

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

def calculate_avg(files,batch):
    cmd = "tail -n 1 "+invalid_path+batch+".txt"
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    num_invalid = int(output.split(':')[-1])
    N = NUM_PER_FILE*len(files)-num_invalid
    avg_img=np.zeros((SIZE,SIZE,3),np.float)
    print str(N)+" valid pics"
    '''
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
    '''
    for f in files:
        label = f[:-4]
        print label
        cmd = "cat "+folder_path+batch+"/"+f
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        entries = output.splitlines()
        for entry in entries:
            e=entry.split()
            idnum=int(e[0])
            '''
            if label in invalid_map and idnum in invalid_map[label]:
                continue
            '''
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
    return N,avg_img

def get_real_avg():
    total=0
    avg_img=np.zeros((SIZE,SIZE,3),np.float)
    cmd = "ls "+avg_path
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    filenames=output.split()
    for f in filenames:
        num=int(f.split('_')[1])
        total+=num
    print total
    for f in filenames:
        img=cv2.imread(avg_path+f,cv2.IMREAD_UNCHANGED).astype(np.float)
        num=int(f.split('_')[1])
        avg_img+=img*(float(num)/float(total))
    cv2.imwrite(avg_path+"real_avg.png",avg_img)

def select_validation_test():
    cmd = "ls "+folder_path
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    filenames=output.split()
    print len(filenames)
    for x in range(200):
        print "start vali round "+str(x)
        vali = open(validation_path+str(x)+".txt", 'w')
        for y in range(1000):
            person_num =  random.randint(0,2622-1)
            person = filenames[person_num][:-4]
            cmd = "cat "+folder_path+person+".txt"
            process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            entries = output.splitlines()
            entry_total = len(entries)
            entry_num =  random.randint(0,entry_total-1)
            entry = entries[entry_num]
            cmd = "sed -i "+str(entry_num+1)+"d "+folder_path+person+".txt"
            process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            vali.write(str(person_num)+" "+person+" "+entry+"\n")
        vali.close()
    print "finished vali"
    for x in range(200):
        print "start test round "+str(x)
        test = open(test_path+str(x)+".txt", 'w')
        for y in range(1000):
            person_num =  random.randint(0,2622-1)
            person = filenames[person_num][:-4]
            cmd = "cat "+folder_path+person+".txt"
            process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            entries = output.splitlines()
            entry_total = len(entries)
            entry_num =  random.randint(0,entry_total-1)
            entry = entries[entry_num]
            cmd = "sed -i "+str(entry_num+1)+"d "+folder_path+person+".txt"
            process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            test.write(str(person_num)+" "+person+" "+entry+"\n")
        test.close()
    print "finished test"

def get_mini_batch():
    batch = list()
    cmd = "ls "+folder_path
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    filenames=output.split()
    for x in range(64):
        person_num = random.randint(0,2622-1)
        person = filenames[person_num][:-4]
        cmd = "cat "+folder_path+person+".txt"
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        entries = output.splitlines()
        entry_total = len(entries)
        entry_num =  random.randint(0,entry_total-1)
        entry = entries[entry_num]
        batch.append(str(person_num)+" "+person+" "+entry)
    return batch


def main():
    select_validation_test()
    '''
    batch=get_mini_batch()
    for b in batch:
        print b
    '''
    #get_real_avg()
    '''
    batch=sys.argv[1]
    total = int(sys.argv[2])
    error = int(sys.argv[3])
    avg_img=cv2.imread(avg_path+batch+"_"+str(total)+"_avg.png",cv2.IMREAD_UNCHANGED)
    print avg_path+batch+"_"+str(total)+"_avg.png"
    avg_img = avg_img.astype(np.float)
    avg_img=avg_img*(float(total)/float(total-error))
    cv2.imwrite(avg_path+batch+"_"+str(total-error)+"_avg.png",avg_img)
    '''
    #files=get_filenames(batch)
    #data=process_files(files,batch)
    #N,avg_img=calculate_avg(files,batch)
    #print "got avg!"
    #cv2.imwrite(avg_path+batch+"_"+str(N)+"_avg.png",avg_img)
    #cv2.imshow('window',avg_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

main()
