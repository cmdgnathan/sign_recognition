import csv
import os
import cv2
import numpy as np

signs = {}
signs['left'] =       0
signs['right'] =      1
signs['stop'] =       2
signs['restricted'] = 3 
signs['target'] =     4



# Write CSV
csv_dir = os.path.abspath( os.path.join( '.','object_frame') )
csv_path = os.path.join( csv_dir, 'csvfile.csv' )

# Extract Image Size
img_path = os.path.abspath( os.path.join( '.','object_frame','0.png') )
print img_path
img = cv2.imread(img_path)
img_w = img.shape[0]
img_h = img.shape[1]

with open(csv_path,'rb') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    for i, line in enumerate(reader):        
        # Ignore Header Line
        if i == 0:
            continue

        image_name, extension = os.path.splitext( os.path.abspath( os.path.join( '.','object_frame',line[0]) ) )
        class_name = line[1]

        x_min = int(line[2])
        y_min = int(line[3])
        w =     int(line[4])
        h =     int(line[5])


        # Current: [x_min] [y_min] [width] [height]
        # YOLOv2: [category number] [object center in X] [object center in Y] [object width in X] [object width in Y]

        x_c = (x_min+w/2.0)/img_w
        y_c = (y_min+h/2.0)/img_h

        x_w = 1.0*w/img_w
        y_h = 1.0*h/img_h

        new_line = [ signs[class_name], x_c, y_c, x_w, y_h]


        txt_path = os.path.join( csv_dir, image_name+'.txt')


        print '==============='
        print 'File:', txt_path
        print 'Line:', new_line

        nl = ' '.join(map(str,new_line))
        with open( txt_path ,'w') as f:
            f.write(nl)
