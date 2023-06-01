import argparse
import glob
import os
import torch
import re
import pandas as pd
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from config import get_evaluate_config
from dataset import data_loader
from eval.post_process import *
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.model.TS_CAN import TSCAN
from neural_methods.model.DeepPhys import DeepPhys
from eval.post_process import *
from collections import OrderedDict
import random
import numpy as np
import matplotlib.pyplot as plt
from signal_methods.POS_WANG import *

RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
g = torch.Generator()
g.manual_seed(RANDOM_SEED)



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def add_args(parser):
    """Adds arguments for parser."""
    parser.add_argument('--config_file', required=False,
                        default="configs/UBFC_TSCAN_EVALUATION.yaml", type=str, help="The name of the model.")
    parser.add_argument(
        '--device',
        default=-1,
        type=int,
        help="An integer to specify which gpu to use, -1 for cpu.")
    parser.add_argument(
        '--model_path', required=False, type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--test_data_path', default='/Users/zhangxiaoyu/Documents/Toolbox/DeepLearningProject/UBFC-Valid/subject1', required=False,
                        type=str, help='The path of the data directory.')
    parser.add_argument('--log_path', default=None, type=str)
    return parser

def load_model(model, config):
    if config.NUM_OF_GPU_TRAIN > 1:
        checkpoint = torch.load(config.INFERENCE.MODEL_PATH)
        state_dict = checkpoint
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.' of dataparallel
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(torch.load(config.INFERENCE.MODEL_PATH))
    model = model.to(config.DEVICE)
    return model

def define_tscan_model(config):
    model = TSCAN(frame_depth=config.MODEL.TSCAN.FRAME_DEPTH, img_size=config.DATA.PREPROCESS.H)
    model = torch.nn.DataParallel(model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
    return load_model(model, config)

def tscan_predict(model, data_loader, config):
    """ Model evaluation on the testing dataset."""
    print(" ====Testing===")
    predictions = dict()
    labels = dict()
    model.eval()
    with torch.no_grad():
        for _, test_batch in enumerate(data_loader):
            subj_index = test_batch[2][0]
            sort_index = int(test_batch[3][0])
            data_test, labels_test = test_batch[0].to(
                config.DEVICE), test_batch[1].to(config.DEVICE)
            N, D, C, H, W = data_test.shape
            data_test = data_test.view(N * D, C, H, W)
            labels_test = labels_test.view(-1, 1)
            data_test = data_test[:(
                                           N * D) // config.MODEL.TSCAN.FRAME_DEPTH * config.MODEL.TSCAN.FRAME_DEPTH]
            labels_test = labels_test[:(
                                               N * D) // config.MODEL.TSCAN.FRAME_DEPTH * config.MODEL.TSCAN.FRAME_DEPTH]
            pred_ppg_test = model(data_test)
            if subj_index not in predictions.keys():
                predictions[subj_index] = dict()
                labels[subj_index] = dict()
            predictions[subj_index][sort_index] = pred_ppg_test
            labels[subj_index][sort_index] = labels_test
    # return np.reshape(np.array(predictions), (-1)), np.reshape(np.array(labels), (-1))
    return predictions, labels

def reform_data_from_dict(data):
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    sort_data = torch.cat(sort_data, dim=0)
    return np.reshape(sort_data.cpu(), (-1))


def inference(confg):
    loader = DataLoader(
        dataset=data_loader.EasyLoader.EasyLoader(name="test", data_path=config.DATA.TEST_DATA_PATH, config_data=config.DATA),
        batch_size=config.INFERENCE.BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker, generator=g)
    model = define_tscan_model(config)
    predictions, labels = tscan_predict(
            model, loader, config)
    key = next(iter(predictions.keys()))
    # Comment: 训练应该只要求了两个信号相关，绝对值和幅度不一定一致，是正常的
    # predcition: 信号序列； label: 标签序列(numpy)
    # gr_pred_hr_fft/peak：根据ground_truth(label)/pred序列，通过傅立叶变换/峰值识别算法得到的心率，是一个数（float）
    prediction = reform_data_from_dict(predictions[key])
    label = reform_data_from_dict(labels[key])
    gt_hr_fft, pred_hr_fft = calculate_metric_per_video(
            prediction, label, fs=config.DATA.FS)
    gt_hr_peak, pred_hr_peak = calculate_metric_peak_per_video(
            prediction, label, fs=config.DATA.FS)
    return prediction,label,gt_hr_fft,pred_hr_fft,gt_hr_peak,pred_hr_peak



    
def load_model(model, config):
    if config.NUM_OF_GPU_TRAIN > 1:
        checkpoint = torch.load(config.INFERENCE.MODEL_PATH)
        state_dict = checkpoint
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.' of dataparallel
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(torch.load(config.INFERENCE.MODEL_PATH))
    model = model.to(config.DEVICE)
    return model


# TODO: change settings here
# 基本上可能用到的都额外列出来了，其他的如有需要可以去改config_file
# 一次只inference 一个subject
parser = argparse.ArgumentParser()
parser = add_args(parser)
parser = data_loader.BaseLoader.BaseLoader.add_data_loader_args(parser)
args = parser.parse_args()
args.test_data_path = '/Users/zhangxiaoyu/Documents/Toolbox/DeepLearningProject/UBFC-Valid/subject3'
args.model_path = '/Users/zhangxiaoyu/Documents/Toolbox/DeepLearningProject/rPPG-Toolbox-1/PreTrainedModels/UBFC_SizeW72_SizeH72_ClipLength180_DataTypeNormalized_Standardized_LabelTypeNormalized/tscan_train_ubfc.pth'

# forms configurations.
config = get_evaluate_config(args)
print('=== Printing Config ===')
print(config)

# Results begin from here
# prediction, label: np.array([])
# gt_hr_fft, pred_hr_fft, gt_hr_peak, pred_hr_peak: float; gt_pred代表是ground_truth还是prediction, fft_peak代表抽取的不同算法
prediction,label,gt_hr_fft,pred_hr_fft,gt_hr_peak,pred_hr_peak = inference(config)
print(gt_hr_peak,pred_hr_peak,';',gt_hr_fft,pred_hr_fft) 

# 传统信号处理方法
video_file = os.path.join(args.test_data_path,'vid.avi')
bvp_file = os.path.join(args.test_data_path,'ground_truth.txt')
signal_prediction, signal_hr =POS_WANG(video_file,bvp_file,'',False)
signal_hr_fft,_ = calculate_metric_per_video(label,signal_prediction,fs=30)
signal_hr_peak,_ = calculate_metric_peak_per_video(label,signal_prediction,fs=30)
print('signal method POS hr:',signal_hr_peak,signal_hr_fft)


# Visualize, 动态和视频同步的可视化可以在script里面做也可以在外头做，这里只是是一边看看。和视频对齐的时候注意prediction的长度和视频的帧数不一定一致，需要对齐
# 可视化除了折线还可以做频谱分析，但我没管
# 可视化的时候如果全部都可视化在一定宽度中会很挤，可以以视频的方式呈现一定时间窗内的信号
# Toy visualization code
x = np.arange(len(prediction))
plt.plot(x, prediction, label='Prediction')
plt.plot(x, label, label='Label')
plt.plot(x, signal_prediction[:-1], label='Signal')
plt.legend()
plt.show()


