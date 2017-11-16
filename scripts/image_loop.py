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


class image_loop:

  def __init__(self):
    #self.image_pub = rospy.Publisher("/usb_cam/image_raw",Image, queue_size=1)
    self.image_pub = rospy.Publisher("/camera/rgb/image_raw",Image, queue_size=1)


    self.image_dir = os.path.abspath( os.path.join( os.path.expanduser('~'),'Projects/darknet/data/actual_signs' ) )
    #self.image_dir = os.path.abspath( os.path.join( os.path.expanduser('~'),'Projects/darknet/data/signs' ) )


    self.image_files = glob.glob( os.path.join( self.image_dir, "*.png" ) )
    self.image_index = 72


    self.bridge = CvBridge()


  def publish(self):

    if self.image_index >= len(self.image_files)-1:
      self.image_index = 0
    else:
      self.image_index += 1

    file_path = self.image_files[self.image_index]

    # Read Image from File
    img = cv2.imread( file_path )

    print "Publishing", file_path

    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
    except CvBridgeError as e:
      print(e)


def main(args):
  ic = image_loop()
  rospy.init_node('image_loop', anonymous=True)

  try:
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
      ic.publish()
      rate.sleep()
  except KeyboardInterrupt:
    print("Shutting down")

  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)