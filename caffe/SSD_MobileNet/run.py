#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.


from mvnc import mvncapi as mvnc
import sys
import numpy
import cv2
import time
import csv
import os
import sys


# ***************************************************************
# Variables
# ***************************************************************
dim    = (300,300)
mean   = 127.5
scale  = 1/127.5

EXAMPLES_BASE_DIR='../../'

# ***************************************************************
# Preprocessor Routines
# ***************************************************************
def center_crop(img):
    dx,dy,dz= img.shape
    delta=float(abs(dy-dx))
    if dx > dy: #crop the x dimension
        img=img[int(0.5*delta):dx-int(0.5*delta),0:dy]
    else:
        img=img[0:dx,int(0.5*delta):dy-int(0.5*delta)]
    return img

def load_preprocess_image(fimg, dim, mean, scale, colorsequence="BGR", centercrop=False):
        # Load the image using open cv which reads the image as BGR
        img_orig = cv2.imread(fimg)
        if img_orig is None:
            print ("ERROR opening ", fimg)
            return None, None

        img = img_orig.astype(numpy.float32)
        
        if centercrop: # Center Crop
            img = center_crop(img)
            
        img = cv2.resize(img, dim) # Resize Image
        if colorsequence == "RGB": # If the colorsequence is RGB, color convert
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 
        img = img - mean           # Subtract Mean
        img = img * scale          # Scale the Image
        return (img_orig, img.astype(numpy.float16))

# ***************************************************************
# get labels
# ***************************************************************
labels = ('background',
          'aeroplane', 'bicycle', 'bird', 'boat',
          'bottle', 'bus', 'car', 'cat', 'chair',
          'cow', 'diningtable', 'dog', 'horse',
          'motorbike', 'person', 'pottedplant',
          'sheep', 'sofa', 'train', 'tvmonitor')

# ***************************************************************
# configure the NCS
# ***************************************************************
mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 2)

# ***************************************************************
# Get a list of ALL the sticks that are plugged in
# ***************************************************************
devices = mvnc.EnumerateDevices()
if len(devices) == 0:
	print('No devices found')
	quit()

# ***************************************************************
# Pick the first stick to run the network
# ***************************************************************
device = mvnc.Device(devices[0])

# ***************************************************************
# Open the NCS
# ***************************************************************
device.OpenDevice()

network_blob='graph'

#Load blob
with open(network_blob, mode='rb') as f:
	blob = f.read()

graph = device.AllocateGraph(blob)

# ***************************************************************
# Load the image
# ***************************************************************
image_file = EXAMPLES_BASE_DIR + "data/images/objdet/TestImage.jpg"
img1, img = load_preprocess_image(image_file, dim, mean, scale)

# ***************************************************************
# Send the image to the NCS
# ***************************************************************
graph.LoadTensor(img.astype(numpy.float16), 'user object')

# ***************************************************************
# Get the result from the NCS
# ***************************************************************
output, userobj = graph.GetResult()

# ***************************************************************
# Print the results of the inference form the NCS
# ***************************************************************
# print(output)
print('---------------------')
for box_index in range(int(output[0])):
    base_index = 7 + box_index * 7
    print (output[base_index:base_index+7])
print('---------------------')


#   a.	First fp16 value holds the number of valid detections = no_valid.
#   b.	The next 6 values are garbage.
#   c.	The next no_valid * 7 values contain the valid detections data in the format: 
#   e.	image_id(always 0 for myriad) | class_id | score | decode_bbox.xmin | decode_bbox.ymin | decode_bbox.xmax | decode_bbox.ymax
img1 = cv2.resize(img1, (700,700))
num_valid_boxes = int(output[0])
print("Number of valid Detections = ", num_valid_boxes)
for ii in range(num_valid_boxes):
        base_index = 7+ ii * 7
        if (numpy.isnan(output[base_index]) or
            numpy.isnan(output[base_index+1]) or
            numpy.isnan(output[base_index+2]) or
            numpy.isnan(output[base_index+3]) or
            numpy.isnan(output[base_index+4]) or
            numpy.isnan(output[base_index+5]) or
            numpy.isnan(output[base_index+6])  ):
            print('skipping box ' + str(ii) + ' because its has NaN')
            continue

        class_id = output[base_index + 1]
        prob = int(output[base_index + 2] * 100)
        disp_txt = labels[int(class_id)] + " (" + str(prob) + "%)"
        print(img1.shape)
        x1 = int(output[base_index+3]*img1.shape[0])
        y1 = int(output[base_index+4]*img1.shape[1])
        x2 = int(output[base_index+5]*img1.shape[0])
        y2 = int(output[base_index+6]*img1.shape[1])
        cv2.rectangle(img1,(x1,y1),(x2,y2),(0,255,0),3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img1,disp_txt,(x1,y1+30), font, 1,(255,255,255),2,cv2.LINE_AA)
        print("Class ID of ", ii , " is = ", labels[int(class_id)], "Confidence = ", output[base_index + 2]*100, "% @(", output[base_index + 3], "," , output[base_index + 4], ") to (", output[base_index + 5], "," , output[base_index + 6], ")")

#order = output.argsort()[::-1][:6]
#print('\n------- predictions --------')
#for i in range(0,5):
#	print ('prediction ' + str(i) + ' (probability ' + str(output[order[i]]) + ') is ' + labels[order[i]] + '  label index is: ' + str(order[i]) )

cv2.imshow("Image", img1)
key = cv2.waitKey(100000) & 0xFF

# ***************************************************************
# Clean up the graph and the device
# ***************************************************************
graph.DeallocateGraph()
device.CloseDevice()
    



