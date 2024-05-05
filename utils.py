import matplotlib.pyplot as plt
import numpy as np
import wave
import scipy.io.wavfile as wavfile
from scipy.signal import firwin, lfilter
import time
from tqdm import tqdm
from pathlib import Path
from vad_utils import read_label_from_file
import os
import torch
#Fs表示初始wav文件的sample rate
def data_type(data):#查看数据类型
    if type(data)==type([1]):
        print("list")
        print(type(data[0]))
        print(len(data))
        if type(data[0])==type(torch.tensor([1])):
           print("tensor!")
           print(data[0].size())
        elif(type(data[0])==type(np.array([1]))):
           print("numpy!")
           print(data[0].shape)
    elif(type(data)==type(torch.tensor([1]))):
        print("tensor")
        print(data.size())
    elif(type(data)==type(np.array([1]))):
        print("numpy")
        print(data.shape)
def plot_wav(wave_data,Fs=16000,show_freq=False):#绘制时域和频域（可选）的图像
    time=np.arange(0,len(wave_data))*1/Fs
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    if show_freq==False:
        #plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
        plt.figure(figsize=(10, 5))
        plt.plot(time,wave_data)
        plt.xlabel('time')
        plt.ylabel('amplitude')
        plt.show()
    else:
        plt.figure(figsize=(10, 10))
        plt.subplot(2,1,1)
        plt.plot(time,wave_data)
        plt.xlabel('time')
        plt.ylabel('amplitude')
        plt.subplot(2,1,2)
        freq=np.abs(np.fft.fft(wave_data))
        plt.plot(np.arange(0,len(freq)),freq)#频率是多少？最高频率是多少？1/Fs?
        plt.xlabel('freq')
        plt.ylabel('amplitude')
        plt.tight_layout()
        plt.show()
def read_wavfile(path="./wavs/dev/54-121080-0009.wav",preEmp=True):#读取wav文件,使用wave module,只返回数据
  with open(path,"rb") as wav_path:
    with wave.open(wav_path, 'rb') as wave_file:

        frame_rate = wave_file.getframerate()#16kHz
        num_frames = wave_file.getnframes()#约为208960数量级            
        #print("帧速率:", frame_rate)
        frames = wave_file.readframes(num_frames)#<class 'bytes'>           
        wave_data = np.frombuffer(frames, dtype = np.short)#int16
        if preEmp==False:#不预加重
            return wave_data
    return preEmphasis(wave_data)
def read_sciwavfile(path="./wavs/dev/54-121080-0009.wav",preEmp=True):#读取wav文件，使用scipy.io.wavfile module
    _, wave_data = wavfile.read(path)
    if preEmp==False:#不预加重
            return wave_data
    return preEmphasis(wave_data)
def test_time(func,rep_time=1):#测试函数效率，repeat time is "rep_time"
    def wrapper(*args, **kw):
       totalTime=0
       for i in tqdm(range(rep_time)):
            s=time.time()
            result=func(*args, **kw)
            e=time.time()
            totalTime+=e-s
       print("function {} total runing time={}".format(func.__name__,totalTime))    
       return result
    return wrapper
def preEmphasis(wav,alpha=0.97):#预加重
    emphasized_signal = np.append(wav[0], wav[1:] - alpha * wav[:-1])
    return emphasized_signal
def zeroCrossingRate(frames):#过零率
    return 0.5*np.sum(np.abs(np.sign(frames[:,:-1])-np.sign(frames[:,1:])),axis=1)/len(frames[0])
def shortTimeEnergy(frames):#短时能量
    return np.sum(np.square(frames),axis=1)/len(frames[0])#sum relates to len(frame)
def shortTimeFreq(frames,Fs=16000):#短时频谱特点：采用加权平均方法
    weight=np.abs(np.fft.rfft(frames,axis=1))/len(frames[0])/len(frames[0])#和长度的平方成正比
    freq=np.fft.rfftfreq(len(frames[0]),1/Fs)
    return np.sum(weight*freq,axis=1)/np.sum(weight)#加权平均
def findPitch(frames,Fs=16000):#找到基频
    min_lag = Fs // 500#自相关函数有延迟
    max_lag = Fs // 20
    pitch_freq=[]
    for i in range(len(frames)):
        autocorr = np.correlate(frames[i], frames[i], mode='full')[len(frames[0]):]
        pitch_freq.append(min_lag+np.argmax(autocorr[min_lag:max_lag]))
    return np.array(pitch_freq,dtype=float)
def extract_features(wav_data, frame_size=0.032, frame_shift=0.008, num_feature=4 ,Fs=16000):#提取特征
    frames_rectangle = wav_to_frames(wav_data, frame_size, frame_shift, 'rectangle')#wav_data就是frames
    frames_hamming = wav_to_frames(wav_data, frame_size, frame_shift, 'hamming')
    zcr = zeroCrossingRate(frames_rectangle)
    ste = shortTimeEnergy(frames_rectangle)
    if num_feature==4:#方法一：四种特征
        stf=shortTimeFreq(frames_hamming, Fs)
        fp = findPitch(frames_hamming, Fs)
        #print(zcr,ste,stf,fp)
        return np.array([100*zcr,np.sqrt(ste),1e2*stf, fp]).T
    else:#方法二：双门限法
        return np.array([100*zcr,np.sqrt(ste)]).T
def wav_to_frames(wav_data,frame_size=0.032,frame_shift=0.008,gate_type='rectangle',Fs=16000):#wav是数组,横坐标为个数,间隔为1/Fs,len/Fs=time
    num_frame_size=int(np.ceil(frame_size*Fs))#一个frame_size中的点数，向上取整
    num_frame_shift=int(np.ceil(frame_shift*Fs))#一个frame_shift中的点数，向上取整
    if gate_type == 'hamming':
        gate = np.hamming((int(num_frame_size)))
    else:
        gate = np.ones(int(num_frame_size))
    start, length = 0, len(wav_data)
    frames = []
    while start < length:
        end = start + num_frame_size
        if end <= length:
            frame = wav_data[start:end]
        else:
            frame = np.pad(wav_data[start:], (0, end - length), 'constant', constant_values=(0, 0))
        frames.append(gate * frame)
        start += num_frame_shift
    return np.array(frames,dtype=float)
def mean_filter(frames, window_size):#均值平滑滤波
    window = np.ones(window_size)/float(window_size)
    return np.convolve(frames, window, 'same')
def lowpass_filter(signal, cutoff_freq=10000, Fs=16000, filter_order=10):#使用FIR滤波器进行低通滤波平滑
    #cutoff_freq: 截止频率（低通滤波器的截止频率）
    #filter_order: 滤波器阶数（默认为10）

    nyquist_freq = 0.5 * Fs    # 计算归一化截止频率
    norm_cutoff_freq = cutoff_freq / nyquist_freq
    filter_coefficients = firwin(filter_order, norm_cutoff_freq)#FIR低通滤波器
    filtered_signal = lfilter(filter_coefficients, 1.0, signal)
    return filtered_signal
def get_features_labels(mode="dev",frame_size=0.032,frame_shift=0.008,preEmp=True,load=True,Fs=16000,batch_num=0):#获取批量的数据和标签（训练模式下）
    #mode can be "dev","test"
    #读wave文件，先wav-frame转化成frames个帧，再提取feature_num个特征
    #得到[np(frames,feature_num=4),np(frames,feature_num=4)......]每个frames长度不一样！
    #使用np.concatenate把它变成二维numpy：numpy(batch*frames,feature_num=4)
    #返回列表，列表元素是二维numpy数组（第二维不相同）[numpy(),numpy()]
    #两个数组分别是features(batch*frames,feature_num=4),labels(batch*frames,label_num=1)
    path="./taskData/"+mode+".npz"
    if os.path.exists(path) and load:#如果已经提取过特征，直接取
        print("load data")
        data=np.load(path,allow_pickle=True)             
        return data["arr_0"],data["arr_1"]#特征,标签 numpy类型
    cwd = Path.cwd()
    if mode in ["dev","train"]:
        label_file=cwd/"data"/(mode+"_label.txt")#dev or train 
        labels_dict=read_label_from_file(path=label_file,frame_size=frame_size,frame_shift=frame_shift)
        file_labels,data_labels=labels_dict.keys(),labels_dict.values()
        wavs_file=cwd/"wavs"/mode
        features,labels=[],[]
        batch_cnt=0
        for data_num,data_label in tqdm(zip(file_labels,data_labels)):#data_num"1110-133-34"类型
            wav_file=wavs_file/(data_num+".wav")
            wav_data=read_sciwavfile(wav_file,preEmp=preEmp)
            feature=extract_features(wav_data,frame_size=frame_size,frame_shift=frame_shift,Fs=Fs)
            features.append(feature)
            batch_cnt+=1
            data_label = np.pad(np.array(data_label), (0, feature.shape[0] - len(data_label)), 'constant', constant_values=(0, 0))
            labels.append(data_label)#[0,0,1,1,1]#每一帧是0或1
            if batch_num!=0 and batch_cnt==batch_num:
                break
        if os.path.exists("./taskData")==False:
           os.makedirs("./taskData")
        print("save data")
        features=np.concatenate(features,axis=0)#变成numpy类型
        labels=np.concatenate(labels,axis=0)
        np.savez(path,features,labels)#存储一个
        return features,labels
    else:#"test mode"
        features=[]
        wavs_file=cwd/"wavs"/mode
        file_nums=[]
        for wav_file in tqdm(wavs_file.iterdir()):
            file_num=wav_file.name.split(".")[0]
            file_nums.append(file_num)
            wav_data=read_sciwavfile(wav_file,preEmp=preEmp)
            feature=extract_features(wav_data,get_feature=get_feature,frame_size=frame_size,frame_shift=frame_shift,Fs=Fs)
            features.append(feature)           
        return file_nums,features
if __name__ =="__main__":
    cwd = Path.cwd()
    dev_wav_path=cwd/"wavs"/"dev"#.iterdir()
    test_data_path=dev_wav_path/"107-22885-0023.wav"
    dev_features,dev_labels=get_features_labels(frame_size=0.025,frame_shift=0.010,batch_num=500)
    data_type(dev_features)
    data_type(dev_labels)
    #print(dev_features.shape)#(batch*frames,point)
    #print(type(dev_features[0]))
    #print(dev_labels.shape)#(batch*frames,point)
    #print(type(dev_labels[0]))
    '''
    path="./taskData"
    np.savez(path+"./task1_data",dev_features,dev_labels)
   
    z=np.load(path+"/task1_data.npz")
    dev_features,dev_labels=z['arr_0'],z["arr_1"]
    print(dev_features.shape)#(batch*frames,point)
    print(type(dev_features[0]))
    print(dev_labels.shape)#(batch*frames,point)
    print(type(dev_labels[0]))
    '''
   
   
