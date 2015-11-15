#!/bin/bash

cd `dirname $0`

${CAFFE_ROOT}/build/tools/caffe train --solver=solver.prototxt
