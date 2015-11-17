"""
utility functions for image processing tools.
"""

import os

import numpy as np
import cv2

import caffe.proto.caffe_pb2 as pb

def load_binaryproto(in_path):
    """
    load binaryproto file(.binaryproto)

    Arguments:
      in_path : path to file reprents mean image(c,h,w)
    Return:
      ndarray represents caffe mean image (c,h,w)
    """
    with open(in_path, "rb") as f:
        b = f.read()

        blob = pb.BlobProto()
        blob.ParseFromString(b)
        data = np.array(blob.data[:], dtype=np.float32)

        #TODO check bug
        img = data.reshape((blob.channels, blob.height, blob.width))

    return img


class Preprocessor(object):
    """
    preprocess image for caffe processing
    """
    def __init__(self, scale=1.0, path_mean_file=None):
        """
        Arguments:
          scale : preprocess scale parameter
          path_mean_file : binaryproto file contains mean image
        """
        # caffe mean image
        self.mean_image = None
        if path_mean_file is not None:
            if not os.path.exists(path_mean_file):
                raise ValueError("%s is not found" % path_mean_file)

            self.mean_image = load_binaryproto(path_mean_file)

        self.scale = scale

    def preprocess(self, src_caffe_image):
        """
        make preprocess image [(src - mean) * mean]
        
        src_caffe_image : caffe image (c, h, w)
        
        Return:
          preprocessed caffe image (c,h,w)
        """

        if self.mean_image is None:
            res = src_caffe_image * self.scale
        else:
            res = (src_caffe_image - self.mean_image) * self.scale
        
        return res
    
class Converter(object):
    """
    Convert image from caffe image to opencv image and vice vaersa.
    """
    def __init__(self, is_gray=False):
        self.is_gray = is_gray

    def to_caffe(self, cv_image):
        """
        from caffe image (c,h,w RGB) to cv image (h,w,c BGR)
        TODO : process gray scale
        """
        m = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR) 
        return m.transpose((2,0,1))   #  (h,w,c) -> (c,h,w)

    def to_cv(self, caffe_image):
        """
        from cv image (h,w,c BGR) to caffe image (c,h,w RGB)
        TODO : process gray scale
        """
        m = np.transpose(caffe_image, axes=(1,2,0))        
        return cv2.cvtColor(m, cv2.COLOR_BGR2RGB) 

        
def make_tiled_image(cv_imgs, width=3, num_tile=9):
    """
    make a tiled image with images.
    
    Arguments:
      cv_imgs : list of images (ndarray shape (h, w, c))
      width : the number of image in a row

    Return:
      cv image
    """
    imgs = cv_imgs
    num_image = len(imgs)

    if len(imgs) == 0:
        return np.zeros((100, 100, 3))
    #print ("num_image : %d" % len(imgs))
    #print ("shape : %s" % str(imgs[0].shape))
    
    data = np.array(imgs)
    padval = 0
    
    # force the number of filters to be square
    padding = ((0, num_tile - num_image), (0, 0), (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    height = data.shape[0] / width

    # tile the filters into an image
    # x,y,h,w,c -> x,h,y,w,c
    data = data.reshape((height, width) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    # x,h,y,w,c ->
    data = data.reshape((height * data.shape[1], width * data.shape[3]) + data.shape[4:])
    
    return data        