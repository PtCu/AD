import numpy as np
from pip import main
from pyparsing import col
from sklearn.utils import check_random_state, check_array
import utilities.utils as utl
import os
import sys
import csv
import pandas as pd

cwd_path = os.getcwd()
data_dir = cwd_path+"/data/"

EXTRA_DATA_NUM = 500

source_file1 = os.path.join(data_dir, "1_NC001_ts.csv")
source_file2 = os.path.join(data_dir, "1_NC002_ts.csv")
source_file3 = os.path.join(data_dir, "1_NC003_ts.csv")
source_file4 = os.path.join(data_dir, "1_NC004_ts.csv")
tmp_file = os.path.join(data_dir, "tmp_data.csv")
dest_file1 = os.path.join(data_dir, "synthetic_data1.csv")
dest_file2 = os.path.join(data_dir, "synthetic_data2.csv")
data = []

SAMPLE_NUM = 170
STEP = 50
GROUP_LABEL = 1
# 正常人标签
NC_TYPE = 0
# 病号标签
PT_TYPE = 1

# title1 = ["ID", "SET", "GROUP"]
# title2 = ["participant_id", "session_id", "diagnosis"]

title1 = ["SET", "GROUP"]
title2 = ["session_id", "diagnosis"]

source_file_list = [source_file1, source_file2, source_file3, source_file4]

session_id = 0


def read_data():
    for idx in range(len(source_file_list)):
        df = pd.read_csv(source_file_list[idx], header=None, sep=',')
        if idx == 0:
            for i in range(df.shape[1]):
                title1.append("ROI")
                title2.append("ROI"+str(i+1))
        
        for i in range(len(df.shape[0])):
            item = []
            # SET
            item.append(int(session_id))
            # GROUP
            if idx == 0:
                item.append(int(NC_TYPE))
            else:
                item.append(int(PT_TYPE))
            for j in range(df.shape[1]):
                item.append(df.at[i, j])

            data.append(item)
      