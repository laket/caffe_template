"""
utils from accessing lmdb
"""

import numpy as np
import caffe.proto.caffe_pb2 as pb
import logging

import lmdb

formatter = logging.Formatter("[%(levelname)s]%(funcName)s(%(lineno)d) %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


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
        
def get_random_keys(path_lmdb, num_key, num_access=1000000):
    """
    randomly get image keys from lmdb

    Args:
      path_lmdb : path to lmdb
      num_key : max number of retrieving key
    Returns:
      list of keys. key[i] = lmdb key of image file
    """
    
    logger.info("start create lmdb random sequence")
    lmdb_env = lmdb.open(path_lmdb, readonly=True, lock=False)

    keys = []

    with lmdb_env.begin() as txn:
        cur = txn.cursor()
        cur.first()
        
        for i in range(num_key):
            k,v = cur.item()

            keys.append(k)

            if not cur.next():
                break

    keys = np.array(keys)
    indices = np.random.permutation(len(keys))[:num_key]

    return list(keys[indices])
