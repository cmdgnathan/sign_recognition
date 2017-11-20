#!/usr/bin/env python
import roslib
roslib.load_manifest('sign_recognition')
import sys
import rospy
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import glob, os



from overlayTemplate import *

import csv

import rospkg

from subprocess import call

from ctypes import *
import math
import random


# DEFINE DARKNET PATH
rospack = rospkg.RosPack()
PACKAGE_PATH = rospack.get_path('sign_recognition')
DARKNET_PATH = rospack.get_path('sign_recognition')+'/darknet'

# CHANGE WORKING DIRECTORY
os.chdir(DARKNET_PATH)

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    return (ctype * len(values))(*values)

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL(DARKNET_PATH+"/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

make_boxes = lib.make_boxes
make_boxes.argtypes = [c_void_p]
make_boxes.restype = POINTER(BOX)

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

num_boxes = lib.num_boxes
num_boxes.argtypes = [c_void_p]
num_boxes.restype = c_int

make_probs = lib.make_probs
make_probs.argtypes = [c_void_p]
make_probs.restype = POINTER(POINTER(c_float))

detect = lib.network_predict
detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

network_detect = lib.network_detect
network_detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.1, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    boxes = make_boxes(net)
    probs = make_probs(net)
    num =   num_boxes(net)
    network_detect(net, im, thresh, hier_thresh, nms, boxes, probs)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if probs[j][i] > 0:
                res.append((meta.names[i], probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_ptrs(cast(probs, POINTER(c_void_p)), num)
    return res


class yolo:

  def __init__(self):

    self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.img_cb)
    #self.image_sub = rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, self.img_cb)
    #self.image_sub = rospy.Subscriber("/raspicam_node/image/compressed", CompressedImage, self.img_cb)    

    self.bridge = CvBridge()

    # Directory Structure
    self.darknet_base = DARKNET_PATH


    # Sign Recognition (General)
    self.darknet_meta = self.darknet_base+'/cfg/signs.data' 
    self.darknet_cfg = self.darknet_base+'/cfg/yolo-sign-full.cfg'
    self.darknet_weights = self.darknet_base+'/signs.weights/yolo-sign-full_800.weights'

    # Sign Recognition (Left / Right)
    # self.darknet_meta = self.darknet_base+'/cfg/signsLR.data' 
    # self.darknet_cfg = self.darknet_base+'/cfg/yolo-sign-full.cfg'
    # self.darknet_weights = self.darknet_base+'/signsLR.weights/yolo-sign-full_800.weights'


    # self.darknet_meta = self.darknet_base+'/cfg/signs9.data' 
    # self.darknet_cfg = self.darknet_base+'/cfg/yolo-sign-full.cfg'
    # self.darknet_weights = self.darknet_base+'/signs9.weights/yolo-sign-full_400.weights'


    self.received = False
    self.loaded = False


    self.img = None

    # HOG
    self.hog = cv2.HOGDescriptor()


    # Darknet Network
    self.net = load_net(self.darknet_cfg, self.darknet_weights, 0)
    self.meta = load_meta(self.darknet_meta)

    # self.net_lr = load_net(self.darknet_cfg_lr, self.darknet_weights_lr, 0)
    # self.meta_lr = load_meta(self.darknet_meta_lr)

    self.loaded = True


    self.image_pub = rospy.Publisher("/usb_cam/image_raw",Image, queue_size=1)

    # Directory of Images
    self.image_dir = os.path.abspath( os.path.join( PACKAGE_PATH,'knn_data' ) )
    self.image_files =  glob.glob( os.path.join( self.image_dir, "*.png" ) )
    self.image_index = 0

    # Store Training Data
    #self.train = np.empty((0,33*25))
    self.train = np.empty((0,34020))
    #self.train = np.empty((0,2*100))
    self.train_labels = np.empty((0))

    # My Sign Dictionary
    self.my_signs = {}
    self.my_signs['left'] =       0
    self.my_signs['right'] =      1
    self.my_signs['stop'] =       2
    self.my_signs['restricted'] = 3 
    self.my_signs['target'] =     4


    # Sign Dictionary
    self.signs = {}
    self.signs['empty'] =      0
    self.signs['left'] =       1
    self.signs['right'] =      2
    self.signs['restricted'] = 3 
    self.signs['stop'] =       4
    self.signs['target'] =     5

    return

  def img_cb(self,data):    


    # Read Image from ROS
    try:
      #self.img = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
      self.img = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    # Raise Flag for Image Reception
    self.received = True


  def cropBBox(self, img, bbox):
    xc, yc, xw, yh = bbox
    x1 = max(0,             int(xc-xw) )
    x2 = min( img.shape[0], int(xc+xw) )
    y1 = max(0,             int(yc-yh) )
    y2 = min( img.shape[1], int(yc+yh) )
    crop = img[ y1:y2,x1:x2,: ]
    return crop

  def maskColor(self, crop, lower, upper):
    # Isolate Color from Background
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)

    hL,sL,vL = lower
    hU,sU,vU = upper

    lower = np.array([hL*255, sL*255, vL*255], dtype="uint8")
    upper = np.array([hU*255, sU*255, vU*255], dtype="uint8")

    mask = cv2.inRange(hsv, lower, upper)

    return mask


  def findBlob(self, mask, crop):
    # Find Largest Blob
    ret,thresh = cv2.threshold(mask, 40, 255, 0)          
    im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)              
        crop = crop[y:(y+h),x:(x+w)]

    # if len(contours) != 0:
    #     # draw in blue the contours that were founded
    #     cv2.drawContours(imgMasked, contours, -1, 255, 3)

    #     #find the biggest area
    #     c = max(contours, key = cv2.contourArea)

    #     x,y,w,h = cv2.boundingRect(c)
    #     # draw the book contour (in green)
    #     cv2.rectangle(imgMasked,(x,y),(x+w,y+h),(0,255,0),2)

    return crop

  def extractFeature(self, img, bbox):
    crop = self.cropBBox( img, bbox )
    mask = self.maskColor( crop, [0.0, 0.3, 0.0], [0.5, 1.0, 1.0] )
    blob = self.findBlob( mask, crop )
    gray = cv2.cvtColor( blob, cv2.COLOR_RGB2GRAY )
    resize = cv2.resize(gray, (128,128), interpolation = cv2.INTER_CUBIC)
    hog = self.hog.compute( resize )
    feature = hog.squeeze()
    return feature

  def detect(self):
    # if self.received and self.loaded:
    #   # Run Darknet    

    #   # Write Image to Darknet Directory
    #   cv2.imwrite( os.path.join( self.darknet_base, 'extracted_frame.png' ), self.img )

    #   r = detect(self.net, self.meta, self.darknet_base+'/extracted_frame.png')

    #   print r

    if self.loaded:
      # Run Darknet    

      ### Load training images and labels
      print "Loading Training Examples for KNN...",      
      with open('../knn_data/test.txt', 'rb') as f:
          reader = csv.reader(f)
          lines = list(reader)


      for i in range(len(lines)):
        file_path = "../knn_data/"+lines[i][0]+".png"

        # Read Image
        img = cv2.imread( file_path )
        img = cv2.resize(img, (300,300), interpolation = cv2.INTER_CUBIC)

        # Run Deep Network
        r = detect(self.net, self.meta, file_path)

        # Store Detections in Dictionary
        detected = {}
        for r_i in range(len(r)):
          sign, conf, bbox = r[r_i]
          detected[sign] = {}
          detected[sign]["conf"] = conf
          detected[sign]["bbox"] = bbox


        # Run KNN on Left/Right Only
        if detected:
        #if "left" in detected.keys() or "right" in detected.keys():

          bbox = detected[sign]["bbox"]
          feature = self.extractFeature(img, bbox)

          # Append Training Features
          self.train = np.vstack( (self.train, feature) ).astype(np.float32)

          # Append Training Labels
          self.train_labels = np.append( self.train_labels, lines[i][1] ).astype(np.int32)

          # plt.figure()
          # plt.subplot(2,1,1)
          # plt.imshow(gray_crop)
          # plt.subplot(2,1,2)
          # plt.imshow(imgMasked)
          # plt.show()



      print "DONE"


      # Training Classifier
      print "Training KNN...",
      self.knn = cv2.ml.KNearest_create()
      self.knn.train(self.train, cv2.ml.ROW_SAMPLE, self.train_labels)
      print "DONE"




      # Loading Test Data
      print "Loading Test Examples...",      
      with open('../knn_data/train.txt', 'rb') as f:
        reader = csv.reader(f)
        lines = list(reader)


      correct = 0.0
      confusion_matrix = np.zeros((6,6))

      for i in range(len(lines)):
        # File Path
        file_path = "../knn_data/"+lines[i][0]+".png"

        # Read Image
        img = cv2.imread( file_path )
        img = cv2.resize(img, (300,300), interpolation = cv2.INTER_CUBIC)

        # Test Label
        test_label = np.int32(lines[i][1])

        # Run Deep Network
        r = detect(self.net, self.meta, file_path)

        # Store Detections in Dictionary
        detected = {}
        for r_i in range(len(r)):
          sign, conf, bbox = r[r_i]
          detected[sign] = {}
          detected[sign]["conf"] = conf
          detected[sign]["bbox"] = bbox


        # Background Detected (Empty Dictionary)
        if not detected:
          ret = self.signs["empty"]
        # Run KNN on Left/Right Only
        else:
        #elif "left" in detected.keys() or "right" in detected.keys():
          bbox = detected[sign]["bbox"]
          feature = self.extractFeature(img, bbox)
          feature = feature.reshape(1,34020)
          ret, results, neighbours, dist = self.knn.findNearest(feature, 5)


        # # Deep Network Classification
        # else:
        #   # Find Most Confident Label
        #   confidence_inverted = [ (detected[keys]["conf"],keys) for keys in detected.keys() ]
        #   max_conf, max_sign = max(confidence_inverted)
          
        #   # Convert My Label to Theirs
        #   ret = self.signs[max_sign]


        print file_path, test_label, ret, test_label == ret

        # Construct Confusion Matrix
        if test_label == ret:
          correct += 1
          confusion_matrix[np.int32(ret)][np.int32(ret)] += 1
        else:
          confusion_matrix[test_label][np.int32(ret)] += 1


      print "DONE"

      print("\n\nTotal accuracy: ", correct/len(lines))
      print(confusion_matrix)






      try:
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
      except CvBridgeError as e:
        print(e)


      self.loaded = False






def main(args):
  ic = yolo()
  rospy.init_node('yolo', anonymous=True)

  try:
    rate = rospy.Rate(2)
    while not rospy.is_shutdown():
      ic.detect()      
      rate.sleep()
  except KeyboardInterrupt:
    print("Shutting down")

  cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
      main(sys.argv)
    except rospy.ROSInterruptException:
      pass
