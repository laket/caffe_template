#!/usr/bin/env python
"""
This shows a graph containing accuracy and loss for each iteration.

Requires:
pandas
matplotlib
"""

import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import argparse

env_key = "CAFFE_ROOT"
if env_key not in os.environ:
    sys.error.write("set CAFFE_ROOT to caffe installed directory.")
    sys.exit(-1)

sys.path.append(os.path.join(os.environ[env_key], "tools/extra"))
from parse_log import parse_log, save_csv_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", dest="path_log", help="path to log file.(ex. /tmp/caffe.INFO)", required=True)
    parser.add_argument("-o", dest="dir_output", help="path to output directory for parsed log files.", required=False)
    args = parser.parse_args()

    path_log, dir_output = args.path_log, args.dir_output

    if not os.path.exists(path_log):
        sys.error.write("specified log file not found.")

    #train_dict_names = ('NumIters', 'Seconds', 'TrainingLoss', 'LearningRate')
    # Seconds : time from run training script (test+train). (time is consumed mostly in train)
    #test_dict_names = ('NumIters', 'Seconds', 'TestAccuracy', 'TestLoss')

    # for old caffe
    #train_dict_list, train_dict_names, test_dict_list, test_dict_names = \
    #    parse_log(path_log)
    train_dict_list, test_dict_list = parse_log(path_log)

    #TODO we should remove dependecy to pandas
    df_train = pd.DataFrame(train_dict_list)
    df_test = pd.DataFrame(test_dict_list)
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # for old caffe
    #lns1 = ax1.plot(df_train["NumIters"], df_train["TrainingLoss"], "g-", label="loss")
    #lns2 = ax2.plot(df_test["NumIters"], 1 - df_test["TestAccuracy"], "r-", label="miss rate")
    lns1 = ax1.plot(df_train["NumIters"], df_train["loss"], "g-", label="loss")
    lns2 = ax2.plot(df_test["NumIters"], 1 - df_test["accuracy"], "r-", label="miss rate")

    lns = lns1 + lns2
    labels = [l.get_label() for l in lns]
    ax1.legend(lns, labels, loc=1)
    
    plt.show()


    if dir_output is not None:
        save_csv_files(path_log, dir_output, train_dict_list,
                       train_dict_names, test_dict_list, test_dict_names)
    


