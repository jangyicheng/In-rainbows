
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline
from IPython import display
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
class Accumulator:
    def __init__(self,n):
        self.data=[0.0]*n
    def add(self,*args):
        self.data=[a+float(b) for a,b in zip(self.data,args)]
    def reset(self):
        self.data=[0.0]*len(self.data)
    def __getitem__(self,idx):
        return self.data[idx]

def use_svg_display():
    backend_inline.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5,2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize']=figsize

def set_axes(axes,xlabel,ylabel,xlim,ylim,xscale,yscale,legend,title=None):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    if legend:
        axes.legend(legend)
    if title:
        axes.set_title(title)
    axes.grid()

class Animator:
    def __init__(self,xlabel=None,ylabel=None,xlim=None,ylim=None,legend=None,
                 xscale='linear',yscale='linear',fmts=('-','m--','g-.','r:'),
                 nrows=1,ncols=1,figsize=(3.5,2.5),title=None):
        if legend is None:
            legend=[]
        use_svg_display()
        self.fig,self.axes=plt.subplots(nrows,ncols,figsize=figsize)
        if nrows*ncols==1:
            self.axes=[self.axes,]
        self.config_axes=lambda:set_axes(self.axes[0],xlabel,ylabel,xlim,ylim,xscale,yscale,legend,title)
        self.X,self.Y,self.fmts=None,None,fmts
    def set_title(self,idx=0,title=None):
        self.axes[idx].set_title(title)
    def add(self,x,y):
        if not hasattr(y,"__len__"):
            y=[y]
        n=len(y)
        if not hasattr(x,"__len__"):
            x=[x]*n
        if not self.X:
           self.X=[[] for _ in range(n)]
        if not self.Y:
           self.Y=[[] for _ in range(n)]
        for i,(a,b) in enumerate(zip(x,y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x,y,fmt in zip(self.X,self.Y,self.fmts):
            self.axes[0].plot(x,y,fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

def accuracy(y_hat,y):#y_hat is a matrix(many types),y is a vector(type),batch
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat=y_hat.argmax(axis=1)
    cmp=y_hat.type(y.dtype)==y
    return float(cmp.type(y.dtype).sum())

# def predict(net,test_iter,n):
#     for X,y in test_iter:
#         break
#     y_hat=net(X)
#     pred=y_hat.argmax(axis=1)
#     def get_fashion_mnist_labels(labels):
#         text_labels=['t-shirt','trouser','pullover','dress','coat',
#                      'sandal','shirt','sneaker','bag','ankle boot']
#         return [text_labels[int(i)] for i in labels]
#     pred=get_fashion_mnist_labels(pred)
#     trues=get_fashion_mnist_labels(y)
#     nrows=n//2
#     ncols=n//nrows
#     show_images(X.reshape(n,224,224),nrows=nrows,ncols=ncols,
#                 titles=["true={},pred={}".format(a,b)
#                         for a,b in zip(trues,np.array(pred))],scale=3)
def train_net(model,features,labels,lr=0.1,momentum=0.9,batch_size=256,num_epoch=10):
    #label应该是[1,0],[0,1]独热向量？
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"train on {device}")
    dataset=TensorDataset(features,labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.BCELoss()#二分类损失函数
    optimizer = optim.SGD(model.parameters(),lr=lr,momentum=momentum)
    animator=Animator(xlabel='epoch',xlim=[1,num_epoch],
    legend=['train acc','train_loss'],title=None)
    for epoch in range(num_epoch):
        metric=Accumulator(3)
        #for batchid,batch in enumerate(dataloader):
        for feature, label in tqdm(dataloader):
            #feature, label=batch
            # print(type(feature))
            # print(label)
            optimizer.zero_grad()
            feature,label=feature.to(device),label.to(device)
            result = model(feature)#result是一个(batch,2)的tensor，分别为不是语音（0）是语音（1）的概率
            loss = criterion(result, label)
            loss.backward()
            optimizer.step()
            metric.add(accuracy(result,label),float(loss.sum()),label.numel())
        train_acc,train_loss=metric[0]/metric[2],metric[1]/metric[2]
        animator.add(epoch+1,(train_acc,)+(train_loss,))
    animator.set_title(title=f"train accuracy rate={train_acc:.4f}")
if __name__ =="__main__":
    x=np.random.randn(10,4)   
    y=np.random.randn(10,1)
    x=torch.from_numpy(x.astype(np.float32))
    y=torch.from_numpy(y.astype(np.float32))
    net = nn.Sequential(nn.Linear(4,1),nn.Sigmoid())
    train_net(net,x,y,batch_size=2)
    print("over")
    #z=TensorDataset(x,y)
    #for i,j in z:
        #print(i,j)
        #print(net(i))