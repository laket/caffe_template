#!/usr/bin/python

"""
get images from lmdb file

Refer
#https://lmdb.readthedocs.org/en/release/
#http://chrischoy.github.io/blog/research/reading-protobuf-db-in-python/

"""

import os
import sys
import logging

import numpy as np

import argparse
import caffe.proto.caffe_pb2 as pb
import cv2
import lmdb

import utils

formatter = logging.Formatter("[%(levelname)s] %(funcName)s(%(lineno)d) : %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

def read_lmdb(path_lmdb, num_data):
    """
    Arguments:
      path_lmdb : path to lmdb directory
      num_data : number of extracted image
    Return:
      list of image(ndarray(OpenCV format))
    """
    
    env = lmdb.open(path_lmdb, readonly=True, lock=False)
    imgs = []
    converter = utils.Converter()
    
    with env.begin() as txn:
        cur = txn.cursor()
        cur.first()

        for i in range(num_data):
            k,v = cur.item()
            datum = pb.Datum()
            datum.ParseFromString(v)

            #print ("w: %d h: %d c: %d" % (datum.width, datum.height,datum.channels))        
            img = np.array(bytearray(datum.data))
        
            img = img.reshape((datum.channels, datum.height, datum.width))
            img = converter.to_cv(img)
            
            #cv2.imwrite("%d.png" % i, img)
            imgs.append(img)

            if not cur.next():
                break
            

    return imgs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest="path_lmdb", help="path to lmdb database to read", required=True)
    parser.add_argument("-o", dest="path_out", help="path to output directory", required=True)
    parser.add_argument("-n", dest="num_data", default=10, type=int, help="the number of extracted images.")
    args = parser.parse_args()
    
    path_lmdb, num_data, path_out = args.path_lmdb, args.num_data, args.path_out


    if not os.path.exists(path_out):
        logger.warning("Create directory %s" % path_out)
        os.makedirs(path_out)
    
    imgs = read_lmdb(path_lmdb, num_data)

    for i, img in enumerate(imgs):
        p = os.path.join(path_out, "%d.png" % i)
        cv2.imwrite(p, img)
    






