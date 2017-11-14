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



class crop:

  def __init__(self):

    #self.image_sub = rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, self.img_cb)
    self.image_sub = rospy.Subscriber("/raspicam_node/image/compressed", CompressedImage, self.img_cb)    

    self.bridge = CvBridge()

    self.num_signs = 1

    # Folder Structure
    self.base_path = os.path.join( os.path.expanduser('~'),'Desktop','signs' )
    self.classes = next(os.walk( os.path.abspath( os.path.join(self.base_path,'crop') )))[1]
    self.frame_number = 0

    # Write CSV
    file_write = os.path.abspath( os.path.join( self.base_path,'object_frame','csvfile.csv') )
    with open(file_write,'wb') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(['image_name','sign','x_min','y_min','width','height'])

    return


  def hist_match(self, source, template):
      """
      Adjust the pixel values of a grayscale image such that its histogram
      matches that of a target image

      Arguments:
      -----------
          source: np.ndarray
              Image to transform; the histogram is computed over the flattened
              array
          template: np.ndarray
              Template image; can have different dimensions to source
      Returns:
      -----------
          matched: np.ndarray
              The transformed output image
      """

      oldshape = source.shape
      source = source.ravel()
      template = template.ravel()

      # get the set of unique pixel values and their corresponding indices and
      # counts
      s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                              return_counts=True)
      t_values, t_counts = np.unique(template, return_counts=True)

      # take the cumsum of the counts and normalize by the number of pixels to
      # get the empirical cumulative distribution functions for the source and
      # template images (maps pixel value --> quantile)
      s_quantiles = np.cumsum(s_counts).astype(np.float64)
      s_quantiles /= s_quantiles[-1]
      t_quantiles = np.cumsum(t_counts).astype(np.float64)
      t_quantiles /= t_quantiles[-1]

      # interpolate linearly to find the pixel values in the template image
      # that correspond most closely to the quantiles in the source image
      interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

      return interp_t_values[bin_idx].reshape(oldshape)

  def img_cb(self,data):

    # Read Image from ROS
    try:
      background = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
      background = cv2.resize(background, (background.shape[0], background.shape[1]))
    except CvBridgeError as e:
      print(e)

    # file_name = os.path.join(os.path.expanduser('~'),'Desktop','frames','frame{}.png'.format(self.frame_number))
    # print cv2.imwrite(file_name,img_bgr8)
    # self.frame_number += 1
    # print 'Wrote:',file_name

    # Choose Random Number of Objects
    random_number = 1#np.random.randint( 0, 1+self.num_signs )

    for n in range(random_number):

      # Choose Random Class (Left,Right,Stop,Restricted,Target)
      random_class = np.random.choice(self.classes)

      # Choose Random Instance of Class
      instance_path, dirs, instances = next(os.walk( os.path.abspath( os.path.join(self.base_path,'crop',random_class) ) ))

      random_instance = np.random.choice(instances)

      # Read Random Instance of Class
      src = cv2.imread( os.path.abspath( os.path.join(instance_path,random_instance) ) )

      # Pad Image
      pad = cv2.copyMakeBorder(src,200,200,200,200,cv2.BORDER_CONSTANT)

      # Randomly Flip Image (Horizontally, Vertically)
      flip_lr = np.random.choice(2,1)
      flip_ud = np.random.choice(2,1)
      if flip_lr:
          pad = np.fliplr(pad)

      if flip_ud:
          pad = np.flipud(pad)

      pad_x,pad_y,pad_c = pad.shape


      # Perspective Distortion
      mu, sigma = 0, 0.001
      r = np.random.normal(mu, sigma, 8)
      #r = np.float32([[r[4],r[0]],[r[5],r[1]],[r[5],r[2]],[r[4],r[3]]])

      pts1 = np.float32([[1,1],[-1,1],[-1,-1],[1,-1]])
      r = np.reshape(r, pts1.shape)
      pts2 = np.float32( np.add(pts1,r) )

      M = cv2.getPerspectiveTransform(pts1,pts2)

      per = cv2.warpPerspective(pad,M,(2000,2000))

      # Flip Perspective
      if flip_lr:
          per = np.fliplr(per)

      if flip_ud:
          per = np.flipud(per)


      # Bounding Box
      gray_per = cv2.cvtColor(per, cv2.COLOR_BGR2GRAY)
      x,y = np.where(gray_per!=0)

      min_x = np.min(x)
      min_y = np.min(y)
      max_x = np.max(x)
      max_y = np.max(y)

      box = per[min_x:max_x,min_y:max_y]

      # Background Integration
      min_size = int(0.2*max(background.shape))
      max_size = int(0.8*max(background.shape))
      #size = np.random.randint(min_size,max_size)
      size = int(np.random.normal(0.4, 0.15, 1)*max(background.shape))
      size = min( max_size, max( min_size, size ) )


      res = cv2.resize(box, (size,int(size*box.shape[0]/box.shape[1])))

      background_raw = background

      background, bbox = overlayTemplate(res, background)





      #ret,overlay_mask = cv2.threshold( cv2.cvtColor(res, cv2.COLOR_BGR2GRAY) ,0,255,cv2.THRESH_BINARY)


    #   # Histogram Matching
    #   res[:,:,0] = self.hist_match( res[:,:,0], background[:,:,0] )    
    #   res[:,:,1] = self.hist_match( res[:,:,1], background[:,:,1] )    
    #   res[:,:,2] = self.hist_match( res[:,:,2], background[:,:,2] )    


    #   # Pad Background
    #   back_pad = cv2.copyMakeBorder(background,overlay_mask.shape[0],overlay_mask.shape[0],overlay_mask.shape[1],overlay_mask.shape[1],cv2.BORDER_CONSTANT)

    #   # Overlay Template on Background
    #   rand_x = int(overlay_mask.shape[0]/2)+np.random.randint(0, background.shape[0]) #normal(int(background.shape[0]/2),int(background.shape[0]/4), 1)
    #   rand_y = int(overlay_mask.shape[1]/2)+int(np.random.normal(background.shape[1]/2,background.shape[1]/8, 1)[0])
    #   back_pad[ rand_x:(rand_x+overlay_mask.shape[0]),rand_y:(rand_y+overlay_mask.shape[1]) ] = \
    #       cv2.add( \
    #           cv2.bitwise_and( res,res, mask=overlay_mask ), \
    #           cv2.bitwise_and( back_pad[ rand_x:(rand_x+overlay_mask.shape[0]),rand_y:(rand_y+overlay_mask.shape[1]) ],back_pad[ rand_x:(rand_x+overlay_mask.shape[0]),rand_y:(rand_y+overlay_mask.shape[1]) ],mask= cv2.bitwise_not(overlay_mask)))

    #   # Crop Background
    #   back_crop = back_pad[overlay_mask.shape[0]:-overlay_mask.shape[0],overlay_mask.shape[1]:-overlay_mask.shape[1]].copy()



    #   background = back_crop

    
    # background = cv2.GaussianBlur(background,(5,5),0)
    # background = cv2.resize(background, (300,300))





    # foreground = res
    # background = cv2.imread(os.path.join(self.base_path,'frames','background.jpg'))




    # hf, wf, cf = foreground.shape
    # hb, wb, cb = background.shape
    # x=np.random.randint(-int(wf/2), int(wb-wf/2))
    # y=np.random.randint(-int(hf/2), int(hb-hf/2))
    
    # # overlay image
    # if x<0:
    #     foreground = foreground[:,-x:,:]
    #     wf=wf+x
    #     x=0
    # elif x>wb-wf:
    #     foreground = foreground[:,:wb-x,:]
    #     wf=wb-x
    # if y<0:
    #     foreground = foreground[-y:,:,:]
    #     hf=hf+y
    #     y=0
    # elif y>hb-hf:
    #     foreground = foreground[:hb-y,:,:]
    #     hf=hb-y
    
    # overlayedImg = np.copy(background)
    # overlayedImg[y:y+hf,x:x+wf,:] = foreground
    
    # # blur border
    # pad_x = int(0.6*wf)
    # pad_y = int(0.6*hf)
    # overlayedImg_padded = cv2.copyMakeBorder(overlayedImg,pad_y,pad_y,pad_x,pad_x,cv2.BORDER_REFLECT)
    # # crop out target patch
    # cropped_patch = overlayedImg_padded[y:y+hf+pad_y*2,x:x+wf+pad_x*2,:]
    # # blur the target patch
    # blurred_patch = cv2.GaussianBlur(cropped_patch, ksize=(0,0), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_REFLECT)
    # # paste back onto the background image
    # overlayedImg_padded[y:y+hf+pad_y*2,x:x+wf+pad_x*2,:] = blurred_patch
    # # crop out border
    # overlayedImg = overlayedImg_padded[pad_y:pad_y+hb, pad_x:pad_x+wb,:]
    # # paste back the original template
    # overlayedImg[y+int(pad_y/2):y+hf-int(pad_y/2),x+int(pad_x/2):x+wf-int(pad_x/2),:] = foreground[int(pad_y/2):hf-int(pad_y/2),int(pad_x/2):wf-int(pad_x/2),:]



    # Write Image Frame to Folder
    file_write = os.path.abspath( os.path.join( self.base_path,'null_frame','{}.png'.format(self.frame_number)) )
    cv2.imwrite(file_write,background_raw)

    file_write = os.path.abspath( os.path.join( self.base_path,'object_frame','{}.png'.format(self.frame_number)) )
    cv2.imwrite(file_write,background)

    # Store Data in Matrix
    #self.file_names.append('{}.png'.format(self.frame_number))


    # Write CSV
    file_write = os.path.abspath( os.path.join( self.base_path,'object_frame','csvfile.csv') )
    with open(file_write,'ab') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(['{}.png'.format(self.frame_number),random_class,bbox[0],bbox[1],bbox[2],bbox[3]])



    self.frame_number += 1

    print "========================"
    print self.frame_number
    print "Wrote:           ",file_write
    print "-- Number:       ", random_number
    if random_number>0:
      print "-- Class:        ", random_class
      print "-- Instance:     ", random_instance
      print "-- Distortion:   "
      print "-- Size:         ", size
      #print "-- Location:     ", rand_x, rand_y
      #print "-- Contrast:     "
    else:
      print "-- Class:        "
      print "-- Instance:     "
      print "-- Distortion:   "
      print "-- Size:         "
      #print "-- Location:     "
      #print "-- Contrast:     "
    if self.frame_number >= 10000:
        rospy.signal_shutdown('frame limit reached')

    # plt.subplot(141),plt.imshow(pad),plt.title('Input')
    # plt.subplot(142),plt.imshow(per),plt.title('Perspective')
    # plt.subplot(143),plt.imshow(box),plt.title('BW')
    # plt.subplot(144),plt.imshow(back_resize),plt.title('BBox')
    # plt.draw()
    # plt.show(block=False)

    # time.sleep(3)
    # plt.close()





def main(args):
  ic = crop()
  rospy.init_node('crop', anonymous=True)

  try:
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
      rate.sleep()
  except KeyboardInterrupt:
    print("Shutting down")

  cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
      main(sys.argv)
    except rospy.ROSInterruptException:
      pass
