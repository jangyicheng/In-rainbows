import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchaudio.compliance import kaldi
import utils

import numpy as np
from sklearn.mixture import GaussianMixture
from python_speech_features import mfcc
import soundfile as sf

feature = kaldi.fbank(wave, sample_frequency=rate, num_mel_bins=40, snip_edges=False)
                
def extract_mfcc(audio, sample_rate):
    # 提取MFCC特征
    mfcc_features = mfcc(audio, samplerate=sample_rate, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=512)
    return mfcc_features

def train_vad_model(features, labels):
    # 训练GMM模型
    gmm = GaussianMixture(n_components=2)
    gmm.fit(features, labels)
    return gmm

def vad_detection(audio, sample_rate, vad_model, threshold):
    # 提取待检测语音片段的MFCC特征
    mfcc_features = extract_mfcc(audio, sample_rate)
    
    # 使用GMM模型进行端点检测
    scores = vad_model.score_samples(mfcc_features)
    decisions = scores[:, 0] - scores[:, 1] >= threshold
    
    # 根据决策结果进行语音端点检测
    speech_segments = []
    is_speech = False
    for i, decision in enumerate(decisions):
        if decision:
            if not is_speech:
                start_frame = i
                is_speech = True
        else:
            if is_speech:
                end_frame = i
                speech_segments.append((start_frame, end_frame))
                is_speech = False
    
    return speech_segments

# 训练数据准备（示例数据）
train_audio, train_sample_rate = sf.read('train.wav')
train_labels = np.array([1, 1, 0, 0])  # 语音活动和非语音活动的标签

# 提取训练数据的MFCC特征
train_features = extract_mfcc(train_audio, train_sample_rate)

# 训练VAD模型
vad_model = train_vad_model(train_features, train_labels)

# 待检测语音数据
test_audio, test_sample_rate = sf.read('test.wav')

# 设置VAD阈值
vad_threshold = 0.5

# 进行语音端点检测
speech_segments = vad_detection(test_audio, test_sample_rate, vad_model, vad_threshold)
class task2Dataset(Dataset):
    def __init__(self, mode='test'):
        save_path = os.path.join('./data', mode + '_data.pt')
        if os.path.exists(save_path):
            self.feature, self.label = torch.load(save_path)
        else:
            data_path = os.path.join('./data', mode + '_label.txt')
            wave_path = os.path.join('./wavs', mode)
            data = utils.read_label_from_file(data_path)
            feature_list, label_list = [], []
            for name, value in tqdm(data.items()):
                path = os.path.join(wave_path, name + '.wav')
                wave, rate = torchaudio.load(path)
                feature = kaldi.fbank(wave, sample_frequency=rate, num_mel_bins=40, snip_edges=False)
                label = F.pad(torch.Tensor(value), (0, feature.shape[0] - len(value)), 'constant', 0)
                feature_list.append(feature)
                label_list.append(label)
            self.feature = torch.cat(feature_list, dim=0)
            self.label = torch.cat(label_list, dim=0)
            torch.save((self.feature, self.label), save_path)

    def __getitem__(self, index):
        return self.feature[index], self.label[index]

    def __len__(self):
        return self.feature.shape[0]


class DNN_Model(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=1):
        super(DNN_Model, self).__init__()
        linear_layer = [
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        ]
        norm_layer = [
            nn.BatchNorm1d(hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        ]
        for layer in linear_layer:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.network = nn.Sequential(
            linear_layer[0],
            norm_layer[0],
            nn.ReLU(True),
            linear_layer[1],
            norm_layer[1],
            nn.ReLU(True),
            linear_layer[2],
            norm_layer[2],
            nn.ReLU(True),
            linear_layer[3],
            nn.Sigmoid()
        )

    def forward(self, inputs):
        outputs = self.network(inputs)
        return outputs