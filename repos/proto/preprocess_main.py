# -*- coding: utf-8 -*-

import os
from mne.io import read_raw_edf
import pandas as pd
import pickle
from tqdm.autonotebook import tqdm

"""
读取.edf原始EEG数据（两种电格式）
.set_eeg_reference()重定位
.filter()滤波范围[l_freq=0.5, h_freq=45.0]
.resample()重采样 200
"""
def load_edf(fname):
    # raw = read_raw_edf(fname, preload=True,encoding="latin1")
    raw = read_raw_edf(fname, preload=True)
    original_raw = raw.copy()
    try:
        original_raw.pick(ch_names_0)
    except:
        original_raw.pick(ch_names_1)

    original_raw.set_eeg_reference('average', projection=False, verbose=False)
    original_raw.filter(l_freq=0.5, h_freq=45.0)  # low-pass filter
    original_raw.resample(200, npad="auto")
    return original_raw

def save_pkl(file, data):
    with open(file, 'wb') as file:
        pickle.dump(data, file)
        file.close()

ch_names_0 = ['EEG Fp1-Ref', 'EEG Fp2-Ref', 'EEG F3-Ref', 'EEG F4-Ref', 'EEG C3-Ref', 'EEG C4-Ref', 'EEG P3-Ref', 'EEG P4-Ref', 'EEG O1-Ref', 'EEG O2-Ref', 'EEG F7-Ref', 'EEG F8-Ref', 'EEG T3-Ref', 'EEG T4-Ref', 'EEG T5-Ref', 'EEG T6-Ref', 'EEG Fz-Ref', 'EEG Cz-Ref', 'EEG Pz-Ref', 'EEG A1-Ref', 'EEG A2-Ref']
ch_names_1 = ['EEG Fp1-REF', 'EEG Fp2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG Fz-REF', 'EEG Cz-REF', 'EEG Pz-REF', 'EEG A1-REF', 'EEG A2-REF']
new_ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', 'A1', 'A2']


source_base = ["/home/lzy/data/脑电/Raw_Data_New/train/轻度",
               "/home/lzy/data/脑电/Raw_Data_New/train/中度",
               "/home/lzy/data/脑电/Raw_Data_New/train/正常",
                "/home/lzy/data/脑电/Raw_Data_New/train/重度",
               "/home/lzy/data/脑电/Raw_Data_New/val/轻度",
               "/home/lzy/data/脑电/Raw_Data_New/val/中度",
               "/home/lzy/data/脑电/Raw_Data_New/val/正常",
                "/home/lzy/data/脑电/Raw_Data_New/val/重度",
               ]

output_base = ["/home/lzy/data/脑电/Processed/raw/train/mild_blue/",
               "/home/lzy/data/脑电/Processed/raw/train/moderate_orange/",
               "/home/lzy/data/脑电/Processed/raw/train/normal_green/",
               "/home/lzy/data/脑电/Processed/raw/train/severe_red/",
               "/home/lzy/data/脑电/Processed/raw/val/mild_blue/",
               "/home/lzy/data/脑电/Processed/raw/val/moderate_orange/",
               "/home/lzy/data/脑电/Processed/raw/val/normal_green/",
               "/home/lzy/data/脑电/Processed/raw/val/severe_red/"
               ]
"""
将原始EEG数据按照T_INTVAL间隔长度进行划分
"""
def transform(source_base, output_base):
    tot_cnt = 0
    fail_cnt = 0
    print(source_base)
    for root, ds, fs in os.walk(source_base):
        if not fs:
            continue
        split_base, category = os.path.split(source_base)
        split = os.path.basename(split_base)
        for f in tqdm(fs, desc=f'{split}/{category}'):
            tot_cnt+=1
            try:
                fullname = os.path.join(root, f)
                original_raw = load_edf(fullname)
                Time_max = original_raw.times.max()
                T_INTVAL = 50
                T_0 = 20
                i = 0
                while T_0 + T_INTVAL < Time_max:
                    new_raw = original_raw.copy().crop(tmin=T_0, tmax=T_0 + T_INTVAL)
                    data, times = new_raw.get_data(return_times=True)
                    dataframe = pd.DataFrame(data)
                    
                    dataframe.index = new_ch_names
                    save_pkl(output_base+'/'+f+'_'+str(i)+".edf",dataframe)
                    T_0 += T_INTVAL
                    i += 1
            except Exception as e:
                print(e)
                print("break")
                fail_cnt+=1
                continue
    print("total_cnt",tot_cnt)
    print("fail_cnt",fail_cnt)

if __name__ == '__main__':

    for i in range(0,8,1):
        transform(source_base[i], output_base[i])
        print(i)