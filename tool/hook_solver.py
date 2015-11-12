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

#suppress caffe output
os.environ["GLOG_minloglevel"] = str(2)

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
    

def train(path_solver, num_iter=100, hook_iter=10):
    """
    Arguments:
      path_solver : path to prototxt of solver
      num_iter : the number of train iteration

    Return:
      trained caffe.Net
    """
    
    solver = caffe.SGDSolver(path_solver)

    for name, blob in solver.net.blobs.items():
        logger.debug("%s[%s]" % (name, str(blob.data.shape)))

    for i in range(num_iter):
        #take 1 step (if iter reaches to test timing or snapshot timing then test or save is done)
        solver.step(i)

        if i % hook_iter == 0:
            hook_train(solver)
        
    return solver.net

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--solver", dest="path_solver", help="path to solver prototxt")
    #parser.add_argument("-q", "--quit", dest="is_quit", default=False, action="store_true")
    args = parser.parse_args()
    #path_solver, suppress_caffe_log = args.path_solver, args.is_quit
    path_solver = args.path_solver

    if not os.path.exists(path_solver):
        logger.error("%s not found" % path_solver)
        sys.exit(-1)

    net = train(path_solver, 100)
