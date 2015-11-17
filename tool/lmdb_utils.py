"""
utils from accessing lmdb
"""

import numpy as np
import caffe.proto.caffe_pb2 as pb

def get_images_with_keys(txn, keys):
    """
    Arguments:
       txn : lmdb.Transaction for lmdb containing Datum
       keys : list of key of lmdb. 

    Return:
       labels[i] : label
       caffe images[i] : ndarray (c,h,w) 
    """
    images = []
    labels = []

    for k in keys:
        v = txn.get(k)

        datum = pb.Datum()
        datum.ParseFromString(v)
        label = datum.label

        img = np.array(bytearray(datum.data))
        img = img.reshape(datum.channels, datum.height, datum.width)

        images.append(img)
        labels.append(label)
        
    return labels, images
        
