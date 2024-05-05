import torch
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from vad_utils import prediction_to_vad_label,read_label_from_file
from evaluate import get_metrics
from utils import get_features_labels,extract_features
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from evaluate import get_metrics
from model_utils import accuracy
#import sklearn.metrics.accuracy_score
frame_size=0.025 #0.025*16000=400个点
frame_shift=0.010 #0.010*16000=160个点
Fs=16000#采样频率，已知
preEmp=True#预加重
model_type="mlp"
batch_num=500
#print(len(dev_features))#三维数组（batch,帧数，四个特征）500*1400*4左右
#print(dev_features[0].shape)
#print(len(dev_labels))#二维数组(batch,帧数)

#print(dev_features.shape)
#print(dev_labels.shape)
def train(dev_features,dev_labels,model_type="svm"):
    #输入类型为numpy
    if model_type=="mlp":
        mlp_clf = LogisticRegression(penalty='l1', solver='liblinear',class_weight='balanced').fit(dev_features,dev_labels)#MLP
        return mlp_clf
    if model_type=="svm":
       #svm_clf = SVC(kernel='linear').fit(dev_features,dev_labels)#SVM 'rbf'
        svm_clf = SVC(probability=True, kernel='linear').fit(dev_features, dev_labels)  # SVM 'rbf'

        return svm_clf
    if model_type=="torch":
        dev_labels=dev_labels[:,np.newaxis]#输入为numpy类型
        dev_features=torch.from_numpy(dev_features.astype(np.float32))#转化为tensor
        dev_labels=torch.from_numpy(dev_labels.astype(np.float32))
        dataset = TensorDataset(dev_features, dev_labels)
        dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
        net = nn.Sequential(nn.Linear(4,2),nn.Sigmoid())#归一化到0-1,输出两个概率：[0.3，0.7]
        criterion = nn.BCELoss()#二分类损失函数
        lr=0.1
        momentum=0.9
        optimizer = optim.SGD(net.parameters(),lr=lr,momentum=momentum)
        for epoch in tqdm(range(10)):
            total_loss=0
            for feature, label in tqdm(dataloader):
                optimizer.zero_grad()
                result = net(feature)#一个概率
                loss = criterion(result, label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()       
            print('epoch: {}, loss: {}'.format(epoch + 1, total_loss))
        return net
def evaluate(model,dev_features,dev_labels,model_type="svm"):
    #输入类型为numpy
    if model_type=="mlp":
        predictions=model.predict(dev_features)
        score=model.score(dev_features,dev_labels)
        print("score=",score)
    if model_type=="svm":
        #predictions=model.predict(dev_features)
        predictions = model.predict_proba(dev_features)  # 返回两个概率
        #print(predictions[0])
    if model_type=="torch":
        dev_features=torch.from_numpy(dev_features.astype(np.float32))
        predictions=model(dev_features)
        predictions=predictions.detach().numpy()
    auc, eer=get_metrics(predictions,dev_labels)
    #TODO acc=accuracy()
    print("auc=",auc)
    print("eer=",eer)   
def predict(model):
    file_num,test_features=get_features_labels("test",frame_size=frame_size,frame_shift=frame_shift)
    model.predict(test_features)
    predictions=[]
    test_label=[]
    for i,prediction in tqdm(enumerate(predictions)):
        label=prediction_to_vad_label(prediction,frame_size=frame_size,frame_shift=frame_shift,threshold=0.5)
        test_label.append(file_num[i]+' '+label+'\n')
    with open("./data/test_label.txt", "w") as file:
        file.write(test_label)
if __name__=="__main__":
    dev_features,dev_labels=get_features_labels("dev",frame_size=frame_size,frame_shift=frame_shift,preEmp=preEmp,batch_num=batch_num)#wavs三维数组(batch,帧数,一帧里点数)
    clf=train(model_type=model_type,dev_features=dev_features,dev_labels=dev_labels)
    evaluate(clf,model_type=model_type,dev_features=dev_features,dev_labels=dev_labels)
        