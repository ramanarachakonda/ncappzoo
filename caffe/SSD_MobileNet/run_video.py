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

dim=(300,300)
EXAMPLES_BASE_DIR='../../'

cv_window_name = "SSD Mobilenet"

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

# handles key presses by adjusting global thresholds etc.
# raw_key is the return value from cv2.waitkey
# returns False if program should end, or True if should continue
def handle_keys(raw_key):
    ascii_code = raw_key & 0xFF
    if ((ascii_code == ord('q')) or (ascii_code == ord('Q'))):
        return False

    return True


def overlay_on_image(image, object_info):
    base_index = 0
    class_id = object_info[base_index + 1]
    prob = int(object_info[base_index + 2] * 100)
    label_text = labels[int(class_id)] + " (" + str(prob) + "%)"
    box_left = int(object_info[base_index + 3] * image.shape[0])
    box_top = int(object_info[base_index + 4] * image.shape[1])
    box_right = int(object_info[base_index + 5] * image.shape[0])
    box_bottom = int(object_info[base_index + 6] * image.shape[1])

    #cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    # draw the rectangle on the image.  This is hopefully around the object
    box_color = (0, 255, 0)  # green box
    box_thickness = 2
    cv2.rectangle(image, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)

    # draw the classification label string just above and to the left of the rectangle
    label_background_color = (70, 120, 70)  # greyish green background for text
    label_text_color = (255, 255, 255)  # white text

    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    label_left = box_left
    label_top = box_top - label_size[1]
    if (label_top < 1):
        label_top = 1
    label_right = label_left + label_size[0]
    label_bottom = label_top + label_size[1]
    cv2.rectangle(image, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1),
                  label_background_color, -1)

    # label text above the box
    cv2.putText(image, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)

    #font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.putText(image, disp_txt, (x1, y1 + 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)



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
    #print(output)

    #   a.	First fp16 value holds the number of valid detections = no_valid.
    #   b.	The next 6 values are garbage.
    #   c.	The next no_valid * 7 values contain the valid detections data in the format:
    #   e.	image_id(always 0 for myriad) | class_id | score | decode_bbox.xmin | decode_bbox.ymin | decode_bbox.xmax | decode_bbox.ymax
    img1 = cv2.resize(img1, (700,700))
    num_valid_boxes = int(output[0])
    #print("Number of valid Detections = ", num_valid_boxes)
    for ii in range(num_valid_boxes):
            base_index = 7+ ii * 7
            if (not numpy.isfinite(output[base_index]) or
                    not numpy.isfinite(output[base_index + 1]) or
                    not numpy.isfinite(output[base_index + 2]) or
                    not numpy.isfinite(output[base_index + 3]) or
                    not numpy.isfinite(output[base_index + 4]) or
                    not numpy.isfinite(output[base_index + 5]) or
                    not numpy.isfinite(output[base_index + 6])):
                # boxes with non infinite (inf, nan, etc) numbers must be ignored
                #print('skipping box ' + str(ii) + ' because its has NaN')
                continue

            overlay_on_image(img1, output[base_index:base_index+7])

            #print("Class ID of ", ii , " is = ", labels[int(class_id)], "Confidence = ", output[base_index + 2]*100, "% @(", output[base_index + 3], "," , output[base_index + 4], ") to (", output[base_index + 5], "," , output[base_index + 6], ")")

    cv2.imshow(cv_window_name, img1)


#cap = cv2.VideoCapture('../../data/videos/Test.MOV')
cap = cv2.VideoCapture('./contrapicado_traffic_shortened_960x540.mp4')

cv2.namedWindow(cv_window_name)

while(cap.isOpened()):
    ret, img1 = cap.read()

    # check if the window is visible, this means the user hasn't closed
    # the window via the X button
    prop_val = cv2.getWindowProperty(cv_window_name, cv2.WND_PROP_ASPECT_RATIO)
    if (prop_val < 0.0):
        break

    runimage(img1)
    raw_key = cv2.waitKey(1)
    if (raw_key != -1):
        if (handle_keys(raw_key) == False):
            end_time = time.time()
            exit_app = True
            break

# ***************************************************************
# Clean up the graph and the device
# ***************************************************************
graph.DeallocateGraph()
device.CloseDevice()

cap.release()
cv2.destroyAllWindows()


