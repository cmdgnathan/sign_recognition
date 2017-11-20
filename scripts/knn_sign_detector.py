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


class sign_detector:

  def __init__(self):

    self.bridge = CvBridge()

    self.received = False
    self.loaded = False


    self.img = None

    # HOG
    self.hog = cv2.HOGDescriptor()


    self.loaded = True

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

      feature = self.extractFeature(img, [0,0,300,300])

      # Append Training Features
      self.train = np.vstack( (self.train, feature) ).astype(np.float32)

      # Append Training Labels
      self.train_labels = np.append( self.train_labels, lines[i][1] ).astype(np.int32)


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

      feature = self.extractFeature(img, [0,0,300,300])
      feature = feature.reshape(1,34020)
      
      ret, results, neighbours, dist = self.knn.findNearest(feature, 5)

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

    self.loaded = False







if __name__ == '__main__':
  ic = sign_detector()
  ic.detect()      
  
  cv2.destroyAllWindows()
