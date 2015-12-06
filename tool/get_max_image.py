#!/usr/bin/env python

"""
This extract images from lmdb which put max values for each layer.

Requirement:
Net has input_dim layer whose name is data like
----------------------
input: "data"
input_dim: 10
input_dim: 3
input_dim: 227
input_dim: 227
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
logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def extract_blob_names(net_def):
    """
    extract blob names which top on convolutional layer and SoftMax Layer.
    
    Arguments
      net_def:  caffe_pb2.NetParameter

    Retrun
      dictionary {layer_name:blob_name}
    """

    blob_names = {}
    
    for layer in net_def.layer:
        name = layer.name
        layertype = layer.type

        if layertype == "Convolution":
            top_name = layer.top[0]
            blob_names[name] = top_name
            
            #top_blob = net.blobs[top_name]
            #logging.debug("%s : shape %s" % (name, str(top_blob.data.shape)))

    return blob_names


def deconvolution_at_filter(net, deconv_blob, idx_filter, image):
    """
    Arguments:
      net : caffe.Net have forwarded 
      deconv_blob : blob name to deconvolution
      idx_filter : index of filter to deconvolution
      image : ndarray (c,h,w RGB)
    Return:
      image (c,h,w RGB)
    """
    prep = utils.Preprocessor.make_from_setting(setting)
    
    net.blobs["data"].data[...] = prep.preprocess(image)

    out = net.forward()

    blob = net.blobs[deconv_blob]

    num_data, num_filter, height, width = blob.data.shape 

    imgs = []
    
    # start deconvolution
    diffs = blob.diff * 0
    diffs[0][idx_filter] = blob.data[0][idx_filter]

    net.deconv_from_layer(deconv_blob, diffs, zero_higher = True)
    
    grad_blob = net.blobs['data'].diff
    grad_blob = grad_blob[0]            # bc01 -> c01
    grad_blob = grad_blob.transpose((1,2,0))    # (c,h,w) -> (h,w,c)
        
    # convert to gray scale (not normal transform)
    grad_img = np.linalg.norm(grad_blob, axis=2)

    vmax, vmin = np.max(grad_img), np.min(grad_img)

    if vmax - vmin < 0.001:
        pass
    else:
        grad_img = (grad_img-vmin)/(vmax-vmin)*255
    grad_img = grad_img.astype(np.uint8)    

    # cv2.GaussianBlur(grad_img, (0,0), 1.0, grad_img)
    # h,w -> h,w,c (BGR)
    color_img = cv2.cvtColor(grad_img, cv2.COLOR_GRAY2RGB)
    # h,w,c -> c, h, w
    color_img = color_img.transpose((2,0,1))

    return color_img
    
def calculate_score_at_each_layer(net, blob_names):
    """
    calculate score(sum of output values) at each target layer.

    Arguments:
      net : caffe.Net whic have already run forward.
      blob_names : target layer dictionary {layer_name:blob_name}.

    Return:
      dictionary {layer_name : score}

    Memo:
      should we use max instead of sum?
    """
    dict_score = {}
    
    for layer_name, blob_name in blob_names.items():
        # net.blobs[blob_name].data[0].shape = (num_output, height, width)
        # score is 1d array : score[i] = i-th filters score
        score = net.blobs[blob_name].data[0].sum(axis=(1,2))
        #score = net.blobs[blob_name].data[0].max(axis=(1,2))
        dict_score[layer_name] = score

    return dict_score    
    
def forward_data(net, net_def, num_top, path_lmdb):
    """
    Return:
       list of images
       return[i][k] : ndarray, combined top-k image(c,h,w) in i-th filter
    """    
    blob_names = extract_blob_names(net_def)

    logger.info("open db %s" % path_lmdb)
    lmdb_env = lmdb.open(path_lmdb, readonly=True, lock=False)

    # key : layer_name value : array of score at each data
    # score is 1d array. score[i] is i-th filters score
    ranking_dict = {}
    lmdb_keys = []
    for layer_name in blob_names.keys():
        ranking_dict[layer_name] = []

    #MAX_ITERATION = 1000000
    MAX_ITERATION = 10000
    logger.info("start read at most %d image files" % MAX_ITERATION)
    prep = utils.Preprocessor.make_from_setting(setting)
    
    with lmdb_env.begin() as txn:
        cur = txn.cursor()
        cur.first()
        
        for i in range(MAX_ITERATION):
            k,v = cur.item()

            lmdb_keys.append(k)
            
            datum = pb.Datum()
            datum.ParseFromString(v)

            #print ("w: %d h: %d c: %d" % (datum.width, datum.height,datum.channels))        
            img = np.array(bytearray(datum.data))
            
            # (w*h,) -> (num_data, channel, height, width)
            img = img.reshape((datum.channels, datum.height, datum.width))

            #net.blobs["data"].data[...] = img
            
            net.blobs["data"].data[...] = prep.preprocess(img)
            out = net.forward()

            scores = calculate_score_at_each_layer(net, blob_names)

            for layer_name, score in scores.items():
                # ranking_dict[layer_name][i] == i-th filters scores (all data's scores)
                ranking_dict[layer_name].append(score)
            
            if not cur.next():
                break

        lmdb_keys = np.array(lmdb_keys)
        logger.info("make top-%d image" % num_top)
        #key: layer_name value : array of score top-k image for each filter.
        # value[i] : top-k images in i-th filter
        top_image_dict = {}

        # scores[i] = i-th data's scores 
        for layer_name, scores in ranking_dict.items():
            scores = np.array(scores)
            # (data, filter) -> (filter, data)
            scores = scores.transpose((1,0))
            rankings = scores.argsort(axis=1)
            rankings = rankings[:, :num_top]
            
            num_filter = rankings.shape[0]
            filter_images = []

            for idx_filter in range(num_filter):
                top_indices = rankings[idx_filter]
                keys = lmdb_keys[top_indices]

                labels, images = lmdb_utils.get_images_with_keys(txn, keys)
                filter_images.append(images)
            
            top_image_dict[layer_name] = filter_images

    return top_image_dict
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", dest="path_out", help="path to output directory", required=True)
    parser.add_argument("--deconv", dest="is_deconv", action="store_true", default=False)
    args = parser.parse_args()
    path_out, is_deconv = args.path_out, args.is_deconv

    #When you use GPU, you should process images with bulk.
    #caffe.set_device(0)
    #caffe.set_mode_gpu()

    net = caffe.Net(setting.path_net, setting.path_weight, caffe.TEST)
    net_def = pb.NetParameter()

    with open(setting.path_net) as f:
        text_format.Merge(f.read(), net_def)    
    
    top_image_dict = forward_data(net, net_def, 9, setting.path_train_lmdb)
    converter = utils.Converter()

    for layer_name, images in top_image_dict.items():
        path_dir = os.path.join(path_out, layer_name)        
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        
        for idx_filter, topk_images in enumerate(images):
            for k, img in enumerate(topk_images):
                if is_deconv:
                    img = deconvolution_at_filter(net, layer_name, idx_filter, img)
                # (c,h,w) ->  (h,w,c)
                cv_image = converter.to_cv(img)
                #cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                topk_images[k] = cv_image
            
            combined_image = utils.make_tiled_image(topk_images, width=3)
            p = os.path.join(path_dir, "%s_%04d.jpg" % (layer_name, idx_filter))
            cv2.imwrite(p, combined_image)
    
