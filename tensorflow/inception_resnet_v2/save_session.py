#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory. 


import numpy as np
import tensorflow as tf

from inception_resnet_v2 import *

slim = tf.contrib.slim

def run(name, image_size, num_classes):
  with tf.Graph().as_default():
    image = tf.placeholder("float", [1, image_size, image_size, 3], name="input")
    with slim.arg_scope(inception_resnet_v2_arg_scope()):
        logits, _ = inception_resnet_v2(image, num_classes, is_training=False, create_aux_logits=False)
    probabilities = tf.nn.softmax(logits, name='output')
    init_fn = slim.assign_from_checkpoint_fn('inception_resnet_v2_2016_08_30.ckpt', slim.get_model_variables('InceptionResnetV2'))

    with tf.Session() as sess:
        init_fn(sess)
        saver = tf.train.Saver(tf.global_variables())
        saver.save(sess, "output/"+name)

run('inception-resnet-v2', 299, 1001)

