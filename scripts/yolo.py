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

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
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

    self.darknet_meta = self.darknet_base+'/cfg/obj.data' 
    self.darknet_cfg = self.darknet_base+'/cfg/yolo-sign-full.cfg'
    self.darknet_weights = self.darknet_base+'/backup/yolo-sign-full_600.weights'

    self.received = False
    self.loaded = False


    self.img = None


    # Darknet Network
    self.net = load_net(self.darknet_cfg, self.darknet_weights, 0)
    self.meta = load_meta(self.darknet_meta)

    self.loaded = True


    self.image_pub = rospy.Publisher("/usb_cam/image_raw",Image, queue_size=1)

    # Directory of Images
    self.image_dir = os.path.abspath( os.path.join( os.path.expanduser('~'),'Projects/darknet/data/actual_signs' ) )
    self.image_files =  glob.glob( os.path.join( self.image_dir, "*.png" ) )
    self.image_index = 0

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



  def detect(self):
    # if self.received and self.loaded:
    #   # Run Darknet    

    #   # Write Image to Darknet Directory
    #   cv2.imwrite( os.path.join( self.darknet_base, 'extracted_frame.png' ), self.img )

    #   r = detect(self.net, self.meta, self.darknet_base+'/extracted_frame.png')

    #   print r

    if self.loaded:
      # Run Darknet    

      if self.image_index >= len(self.image_files)-1:
        self.image_index = 0
      else:
        self.image_index += 1

      file_path = self.image_files[self.image_index]

      r = detect(self.net, self.meta, file_path)


      # Read Image from File
      img = cv2.imread( file_path )

      img = cv2.resize(img, (300,300), interpolation = cv2.INTER_CUBIC)

      print "Publishing", file_path, r

      try:
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
      except CvBridgeError as e:
        print(e)






def main(args):
  ic = yolo()
  rospy.init_node('yolo', anonymous=True)

  try:
    rate = rospy.Rate(0.5)
    while not rospy.is_shutdown():

      print ic.loaded
      print ic.received

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
