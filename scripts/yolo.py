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
import os

from overlayTemplate import *

import csv

import rospkg

from subprocess import call


class yolo:

  def __init__(self):

    self.image_sub = rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, self.img_cb)
    #self.image_sub = rospy.Subscriber("/raspicam_node/image/compressed", CompressedImage, self.img_cb)    

    self.bridge = CvBridge()

    # Directory Structure
    self.darknet_base = os.path.join( os.path.expanduser('~'),'Projects','darknet' )
    self.darknet_exe = './darknet'
    self.darknet_cfg_data = 'cfg/obj.data' 
    self.darknet_cfg_obj = 'cfg/yolo-obj.2.0.cfg' 
    self.darknet_weights = 'yolo-obj_300.weights'

    self.received = False

    return


  def img_cb(self,data):

    # Read Image from ROS
    try:
      img = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    # Write Image to Darknet Directory
    cv2.imwrite( os.path.join( self.darknet_base, 'extracted_frame.png' ), img )

    # Raise Flag for Image Reception
    self.received = True



  def detect(self):
    if self.received:
      # Run Darknet

      call([self.darknet_exe, \
        'detector', \
        'test', \
        self.darknet_cfg_data, \
        self.darknet_cfg_obj, \
        self.darknet_weights ], \
        cwd=self.darknet_base)





def main(args):
  ic = yolo()
  rospy.init_node('yolo', anonymous=True)

  try:
    rate = rospy.Rate(0.1)
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
