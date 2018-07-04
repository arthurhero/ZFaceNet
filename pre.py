# import the necessary packages
import numpy as np
import urllib2
import cv2
 
def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    try:
        resp = urllib2.urlopen(url)
    except Exception:
        print "error"
        return np.array([])

    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def crop_and_scale(img,left,top,right,bottom):
    top=int(round(top))
    left=int(round(left))
    right=int(round(right))
    bottom=int(round(bottom))
    crop_img = img[top:bottom, left:right]
    scale_img=cv2.resize(crop_img,(256,256))
    return scale_img


def main():
    img=url_to_image("http://images.starpulse.com/news/bloggers/24/blog_images/ashlee-simpson-and-evan-ross.jpg")
    if img.shape==(0,):
        return
    print img.shape
    img=crop_and_scale(img,122.26,170.76,339.52,388.02)
    print img.shape
    cv2.imshow("Image", img)
    cv2.waitKey(0)

main()
