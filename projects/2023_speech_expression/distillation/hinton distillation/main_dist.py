"""
@author: Jiaxin Ye
@contact: jiaxin-ye@foxmail.com
"""
# -*- coding:UTF-8 -*-
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from Model_dist import TIMNET_Model
import argparse
import time
import datetime

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default="test")
parser.add_argument('--model_path', type=str, default='../Models/')
parser.add_argument('--result_path', type=str, default='../Results/')
parser.add_argument('--test_path', type=str, default='./RAVDE_46')
parser.add_argument('--data', type=str, default="./RAVDESS_mfcc_60")
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--beta1', type=float, default=0.93)
parser.add_argument('--beta2', type=float, default=0.98)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epoch', type=int, default=300)####epoch300
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--random_seed', type=int, default=46)
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--filter_size', type=int, default=39)####nb_filters39
parser.add_argument('--dilation_size', type=int, default=8)####dilation8
parser.add_argument('--kernel_size', type=int, default=2)####kernel_size2
parser.add_argument('--stack_size', type=int, default=1)
parser.add_argument('--split_fold', type=int, default=10)
#parser.add_argument('--gpu', type=str, default='1')


args = parser.parse_args()

if args.data=="IEMOCAP" and args.dilation_size!=10:
    args.dilation_size = 10


#os.environ['CUDA_VISIBLE_DEVICES'] = "1"
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print("@@@",gpus[0])
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)
print(f"###gpus:{gpus}")

CLASS_LABELS_finetune = ("angry", "fear", "happy", "neutral","sad")                                   #5
CASIA_CLASS_LABELS = ("angry", "fear", "happy", "neutral", "sad", "surprise")#CASIA                   #6
EMODB_CLASS_LABELS = ("angry", "boredom", "disgust", "fear", "happy", "neutral", "sad")#EMODB         #7
SAVEE_CLASS_LABELS = ("angry","disgust", "fear", "happy", "neutral", "sad", "surprise")#SAVEE         #7
RAVDE_CLASS_LABELS = ("angry", "calm", "disgust", "fear", "happy", "neutral","sad","surprise")#rav    #8
IEMOCAP_CLASS_LABELS = ("angry", "happy", "neutral", "sad")#iemocap                                   #4
EMOVO_CLASS_LABELS = ("angry", "disgust", "fear", "happy","neutral","sad","surprise")#emovo           #7
CLASS_LABELS_dict = {"CASIA": CASIA_CLASS_LABELS,
               "EMODB": EMODB_CLASS_LABELS,
               "EMOVO": EMOVO_CLASS_LABELS,
               "IEMOCAP": IEMOCAP_CLASS_LABELS,
               "RAVDE": RAVDE_CLASS_LABELS,
               "SAVEE": SAVEE_CLASS_LABELS}

with tf.device('/device:GPU:0'):
    # path = "C:\\Users\\711_2\\Desktop\\Yuna_Hong\\speech_expression\\TIM-Net_SER-main\\TIM-Net_SER-main\\Code\\RAVDESS_mfcc_80.npy"
    data = np.load(args.data + ".npy", allow_pickle=True).item()
    # data = np.load(path,allow_pickle=True).item()
    print(np.shape(data["x"]))
    x_source = data["x"]
    y_source = data["y"]
    CLASS_LABELS = CLASS_LABELS_dict['RAVDE']

    model = TIMNET_Model(dilation_size=8, filter_size= 39, args=args, input_shape=x_source.shape[1:], class_label=CLASS_LABELS)
    start = time.time()  # 시작 시간 저장
    if args.mode == "train":
        model.train(x_source, y_source)
    elif args.mode == "test":
        x_feats, y_labels = model.test(x_source, y_source,
                                       path=args.test_path)  # x_feats and y_labels are test datas for t-sne

    end = time.time()
    sec = (end - start)
    result = datetime.timedelta(seconds=sec)
    print(result)