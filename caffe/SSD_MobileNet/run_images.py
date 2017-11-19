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
from os import listdir, system, getpid
from os.path import isfile, join

dim=(300,300)
EXAMPLES_BASE_DIR='../../'

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
def preprocess(src):
    img = cv2.resize(src, (300,300))
    img = img - 127.5
    img = img * 0.007843
    return img

#img1 = cv2.imread(EXAMPLES_BASE_DIR+'data/images/TestImage.jpg')


def runimage(img1):
        img = preprocess(img1)

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
        print(output)
        
        #   a.	First fp16 value holds the number of valid detections = no_valid.
        #   b.	The next 6 values are garbage.
        #   c.	The next no_valid * 7 values contain the valid detections data in the format: 
        #   e.	image_id(always 0 for myriad) | class_id | score | decode_bbox.xmin | decode_bbox.ymin | decode_bbox.xmax | decode_bbox.ymax
        img1 = cv2.resize(img1, (700,700))
        num_valid_boxes = int(output[0])
        print("Number of valid Detections = ", num_valid_boxes)
        for ii in range(num_valid_boxes):
                base_index = 7+ ii * 7
                class_id = output[base_index + 1]
                prob = int(output[base_index + 2] * 100)
                disp_txt = labels[int(class_id)] + " (" + str(prob) + "%)"
                x1 = int(output[base_index+3]*img1.shape[0])
                y1 = int(output[base_index+4]*img1.shape[1])
                x2 = int(output[base_index+5]*img1.shape[0])
                y2 = int(output[base_index+6]*img1.shape[1])
                cv2.rectangle(img1,(x1,y1),(x2,y2),(0,255,0),3)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img1,disp_txt,(x1,y1+30), font, 1,(255,255,255),2,cv2.LINE_AA)
                #print("Class ID of ", ii , " is = ", labels[int(class_id)], "Confidence = ", output[base_index + 2]*100, "% @(", output[base_index + 3], "," , output[base_index + 4], ") to (", output[base_index + 5], "," , output[base_index + 6], ")")

        cv2.imshow("Image", img1)
        key = cv2.waitKey(1000) & 0xFF


images_folder = "../../data/images/objdet/"
imgarr = []
imgarr_orig = []
onlyfiles = [f for f in listdir(images_folder) if isfile(join(images_folder, f))]

# Only load the first 250 files, 
# so that we don"t exceed availible resources
onlyfiles = onlyfiles[:250]

print("Found ", len(onlyfiles), " images")
image_ext_list = [".jpg", ".png", ".JPEG", ".jpeg", ".PNG", ".JPG"]
# Preprocess images
for file in onlyfiles:
    fimg = images_folder + file
    if any([x in image_ext_list for x in fimg]):
        print(fimg + " is not an image file")
        continue
    img = cv2.imread(fimg)
    if img is None:
        print ("ERROR opening ", fimg)
        continue
    runimage(img)

# ***************************************************************
# Clean up the graph and the device
# ***************************************************************
graph.DeallocateGraph()
device.CloseDevice()


