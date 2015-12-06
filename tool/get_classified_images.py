#!/usr/bin/env python

"""
get images miss-classified by a specified caffe model.

this makes tiled images with correct images and wrong images.

Requirement:
Net must has input layer and output "prob"
----------------------
input: "data"
input_dim: 1
input_dim: 3
input_dim: 227
input_dim: 227


layer {
  name: "prob"
  type: "Softmax"
  bottom: "ip1"
  top: "prob"

  include {
    phase: TEST
  }
}
----------------------
"""

import os
import argparse
import logging

import numpy as np
import cv2
import lmdb

os.environ["GLOG_minloglevel"] = str(2)

import caffe
import caffe.proto.caffe_pb2 as pb
from google.protobuf import text_format

import tool_setting as setting
import utils
import lmdb_utils

formatter = logging.Formatter("[%(levelname)s]%(funcName)s(%(lineno)d) %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

    

def classify_lmdb_images(net, path_lmdb, preprocessor, num_access_data=1000):
    """
    Arguments:
      net : caffe Classifier
      path_lmdb : image source
      preprocessor : utils.Preprocessor
    
    Returns:
      labels : true labels
      images : images    
      detect_labels : labels detected by caffe net
    """

    keys = lmdb_utils.get_random_keys(path_lmdb, num_access_data)
    
    logger.info("reading %d image files from %s" % (num_access_data, path_lmdb))    
    lmdb_env = lmdb.open(path_lmdb, readonly=True, lock=False)

    with lmdb_env.begin() as txn:
        labels, images = lmdb_utils.get_images_with_keys(txn, keys)
    logger.info("finished reading.")
    
    detect_labels = []
    logger.info("classifying images...")

    for label, image in zip(labels, images):
        #net.blobs["data"].data[...] = img
        net.blobs["data"].data[...] = preprocessor.preprocess(image)
        out = net.forward()
        probs = net.blobs["prob"].data
        detect_label = np.argmax(probs[0])

        detect_labels.append(detect_label)
            
    return labels, images, detect_labels

def divide_images(labels, detect_labels, images):
    """
    Returns:
      corrects : {true label : [(detect label, caffe images classified correctly)]}
      wrongs :   {true label : [(detect label, caffe images classified wrongly)]}     
    """
    all_labels = labels + detect_labels
    categories = list(np.unique(all_labels))
    num = len(categories)

    corrects = {}
    wrongs = {}
    for c in categories:
        corrects[c] = []
        wrongs[c] = []

    for label, detect_label, image in zip(labels, detect_labels, images):
        if label == detect_label:
            corrects[label].append((detect_label, image))
        else:
            wrongs[label].append((detect_label, image))

    return corrects, wrongs
    
def stat_classification(labels, detect_labels):
    all_labels = labels + detect_labels
    categories = list(np.unique(all_labels))
    num = len(categories)
    
    stats = np.zeros((num, num), dtype=np.int32)

    print categories
    for correct, detect in zip(labels, detect_labels):
        c = categories.index(correct)
        d = categories.index(detect) 
        
        stats[c][d] += 1

    print stats
    return categories, stats
        

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("-o", dest="path_out", help="path to output directory", required=True)
    args = parser.parse_args()
    #path_out, is_deconv = args.path_out, args.is_deconv

    #When you use GPU, you should process images with bulk.
    #caffe.set_device(0)
    #caffe.set_mode_gpu()

    net = caffe.Net(setting.path_net, setting.path_weight, caffe.TEST)

    preprocessor = utils.Preprocessor(setting.scale, setting.path_mean)
    labels, images, detect_labels = classify_lmdb_images(net, setting.path_test_lmdb, preprocessor, num_access_data=1000)

    categories, stats = stat_classification(labels, detect_labels)
    corrects, wrongs = divide_images(labels, detect_labels, images)
    
    converter = utils.Converter()

    idx_selected = 0
    idx_image = 0
    num_image = 9

    cv2.namedWindow("correct", cv2.WINDOW_NORMAL)
    cv2.namedWindow("wrong", cv2.WINDOW_NORMAL)    
    
    while True:
        label = categories[idx_selected]
        
        c_images = zip(*corrects[label])[1][idx_image:idx_image+num_image]
        w_images = zip(*wrongs[label])[1][idx_image:idx_image+num_image]

        c_images = [converter.to_cv(c) for c in c_images]
        w_images = [converter.to_cv(c) for c in w_images]

        tile_correct = utils.make_tiled_image(c_images)
        tile_wrong = utils.make_tiled_image(w_images)

        font = cv2.FONT_HERSHEY_PLAIN
        font_scale = 1.0
        text_color = (0,0,255)
        cv2.putText(tile_correct,"label:%d" % label,(10,10), font, font_scale,text_color)
        cv2.imshow("correct", tile_correct)
        cv2.imshow("wrong", tile_wrong)
        
        keycode = cv2.waitKey(60) & 0xFF
        if keycode == ord("q"):
            break
        if keycode == ord("j"):
            idx_image = max(idx_image-num_image,0)
        if keycode == ord("l"):
            idx_image = idx_image + num_image
        if keycode == ord("i"):
            idx_selected = max(idx_selected-num_image,0)
            idx_image = 0
        if keycode == ord("k"):
            idx_selected = min(idx_selected + 1, len(categories) - 1)
            idx_image = 0
