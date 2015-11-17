#!/usr/bin/env python

"""
sample script to learn model by python.
You can modify this to hook anything with learning process.

call this script in a directory which contains model file.
"""

import os
import sys
import argparse
import logging

import caffe.proto.caffe_pb2 as pb
from google.protobuf import text_format

#suppress caffe output
#os.environ["GLOG_minloglevel"] = str(2)

import caffe

formatter = logging.Formatter("[%(levelname)s]%(funcName)s(%(lineno)d) : %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

caffe.set_device(0)
caffe.set_mode_gpu()

path_model = "net.prototxt"
path_temporary_model = "temporary_net.prototxt"

def hook_train(solver):
    """
    hook process in training

    Arguments:
      solver : caffe.Solver
    """
    test_net = solver.test_nets[0]
    test_net.forward()

    acc = test_net.blobs["accuracy"].data
    logger.info("acc = %s" % str(acc))

    return

def override_net_def(params):
    net_def = pb.NetParameter()
    
    with open(path_model) as f:
        text_format.Merge(f.read(), net_def)    

    dict_layer = {}
    for layer in net_def.layer:
        name = layer.name
        dict_layer[name] = layer

    co = params["CONV_OUTPUT"]
    dict_layer["conv1"].convolution_param.num_output = co[0]
    dict_layer["conv2"].convolution_param.num_output = co[1]
    dict_layer["conv3"].convolution_param.num_output = co[2]


    # in previous version kernel_size is not repeated.
    # so how to set value is changed
    cs = params["CONV_SIZE"]
    dict_layer["conv1"].convolution_param.kernel_size.append(cs[0])
    dict_layer["conv2"].convolution_param.kernel_size.append(cs[1])
    dict_layer["conv3"].convolution_param.kernel_size.append(cs[2])


    print str(net_def)
    with open(path_temporary_model, "w") as f:
        f.write(str(net_def))
        

def train(params, path_solver, num_iter=100, hook_iter=10):
    """
    Arguments:
      path_solver : path to prototxt of solver
      num_iter : the number of train iteration

    Return:
      trained caffe.Net
    """
    override_net_def(params)
    solver = caffe.SGDSolver(path_solver)

    solver.step(num_iter)
        
    return solver.test_nets[0]

# Write a function like this called 'main'
def main(job_id, params):
    path_solver = "/data/cifar10/model/cifar_spearmint/solver.prototxt"

    if not os.path.exists(path_solver):
        logger.error("%s not found" % path_solver)
        sys.exit(-1)

    # 80 -> 3200, 
    test_net = train(params, path_solver, num_iter=50000, hook_iter=None)

    scores = []
    # run 1 batch
    # TODO we need multiple forward to save memory
    for i in range(300):
        test_net.forward()
        scores.append(test_net.blobs["accuracy"].data)

    acc = numpy.average(scores)
    #logger.info("acc = %s" % str(acc))

    return 1 - acc
