import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import scale
np.set_printoptions(threshold=10000000)
#data_processing
rootdir=r'D:\pycharm_community\python_workspace\fc_coach\data'
list=os.listdir(rootdir)
total_data=[]
i=1
for dir in list:
    dir_list=os.path.join(rootdir,dir)
    path_list=os.listdir(dir_list)
    file_name=os.path.join(dir_list,path_list[0])
    data=pd.read_excel(file_name, sheet_name='Sheet1')
    feature=data.groupby(['latitude','longitude','section','km/h','meter']).count()
    feature1=feature.index
    feature2=feature1.values.tolist()
    features=np.array(feature2)
    m,n=features.shape
    label=i*np.ones([m,1])
    label_data=np.concatenate([label,features],axis=1)
    total_data.append(label_data)
    i+=1
total_data=np.array(total_data)
data=[]
for i in range(total_data.shape[0]):
    for j in range(total_data[i].shape[0]):
        data.append(total_data[i][j])
shuffle_num= np.random.permutation(len(data))
data=np.array(data)[shuffle_num]  #带标签数据集

#将特征和标签分开
train=[]
train_label=[]
test=[]
test_label=[]
for i in range(data.shape[0]): #按9：1划分数据集
    if i<(9*data.shape[0]//10):
        train.append(data[i][1:len(data[i])])
        if data[i][0]==1:
            train_label.append([1,0,0,0,0,0,0,0])
        elif data[i][0]==2:
            train_label.append([0,1, 0, 0, 0, 0, 0, 0])
        elif data[i][0] == 3:
            train_label.append([0, 0, 1, 0, 0, 0, 0, 0])
        elif data[i][0] == 4:
            train_label.append([0, 0, 0, 1, 0, 0, 0, 0])
        elif data[i][0] == 5:
            train_label.append([0, 0, 0, 0, 1, 0, 0, 0])
        elif data[i][0] == 6:
            train_label.append([0, 0, 0, 0, 0, 1, 0, 0])
        elif data[i][0] == 7:
            train_label.append([0, 0, 0, 0, 0, 0, 1, 0])
        elif data[i][0] == 8:
            train_label.append([0, 0, 0, 0, 0, 0, 0, 1])
    else:
        test.append(data[i][1:len(data[i])])
        if data[i][0]==1:
            test_label.append([1,0,0,0,0,0,0,0])
        elif data[i][0]==2:
            test_label.append([0,1, 0, 0, 0, 0, 0, 0])
        elif data[i][0] == 3:
            test_label.append([0, 0, 1, 0, 0, 0, 0, 0])
        elif data[i][0] == 4:
            test_label.append([0, 0, 0, 1, 0, 0, 0, 0])
        elif data[i][0] == 5:
            test_label.append([0, 0, 0, 0, 1, 0, 0, 0])
        elif data[i][0] == 6:
            test_label.append([0, 0, 0, 0, 0, 1, 0, 0])
        elif data[i][0] == 7:
            test_label.append([0, 0, 0, 0, 0, 0, 1, 0])
        elif data[i][0] == 8:
            test_label.append([0, 0, 0, 0, 0, 0, 0, 1])


innum=5
hidnum1=10
hidnum2=12
outnum=8
#初始化参数
w1=np.random.randint(-1,1,(innum, hidnum1)).astype(np.float64)  #5x10
b1=np.random.randint(-1,1,(1,hidnum1)).astype(np.float64)       #1x10
w2=np.random.randint(-1,1,(hidnum1, hidnum2)).astype(np.float64)  #10x12
b2=np.random.randint(-1,1,(1,hidnum2)).astype(np.float64)  #1x12
w3=np.random.randint(-1,1,(hidnum2,outnum)).astype(np.float64)  #12x8
b3=np.random.randint(-1,1,(1,outnum)).astype(np.float64)  #1x8

loopNumber=1#epoch次数
xite=0.01
accuracy=[]
E=[]
FI=[]
dw1=np.zeros((innum,hidnum1),dtype=np.float64) #权重导数
db1=np.zeros((1,hidnum1),dtype=np.float64)#偏差导数
dw2=np.zeros((hidnum1,hidnum2),dtype=np.float64) #权重导数
db2=np.zeros((1,hidnum2),dtype=np.float64)#偏差导数
dw3=np.zeros((hidnum2,outnum),dtype=np.float64) #权重导数
db3=np.zeros((1,outnum),dtype=np.float64)#偏差导数
for j in range(loopNumber):
    for i in range(len(train)//3):
        input_data=np.mat(train[i]).astype(np.float64)  #1x5
        pre_output=np.mat(train_label[i]).astype(np.float64) #1x8
        h=(np.dot(input_data,w1)+b1).astype(np.float64)  #1x10
        print('未激活h:',h)
        h_out=(1/(1+np.exp(-h))).astype(np.float64) #sigmoid激活函数
        print('激活h:',h_out)
        h1=(np.dot(h_out,w2)+b2).astype(np.float64)  #1x12
        h1_out=(1/(1+np.exp(-h1))).astype(np.float64) #sigmoid激活函数
        y=(np.dot(h1_out,w3)+b3).astype(np.float64)  #1x8
        y_out=(1/(1+np.exp(-y))).astype(np.float64) #输出层输出
        print('输出:',y_out)

        #计算误差并储存
        e=pre_output-y_out #1x8
        print('误差：',e)
        E.append(sum(np.abs(e)))#储存损失值画损失函数
        #计算权值变化率
        db3=e
        dw3=np.multiply(e,y_out)
        w3=w3+xite*dw3
        b3=b3+xite*db3
        FI=np.multiply(h1_out,(1-h1_out)) #1x12
        e1=np.multiply(np.dot(w3,e.T).T,FI) #1*12
        db2=e1
        dw2=np.multiply(e1,h1_out)
        w2=w2+xite*dw2
        b2=b2+xite*db2
        FJ = np.multiply(h_out, (1-h_out))
        e2=np.multiply(np.dot(w2,e1.T).T,FJ)
        db1=e2
        dw1=np.multiply(e2,h_out)
        #更新权重和偏差
        w1=w1+xite*dw1
        b1=b1+xite*db1

        #储存准确率，以备画准确率曲线
        result = []
        for i in range(len(test)):
            input_data = np.mat(test[i]).astype(np.float64)
            pre_output=np.mat(test_label[i]).astype(np.float64)
            h=np.dot(input_data,w1)+b1
            h_out=1/(1+np.exp(-h)) #sigmoid激活函数
            h1=np.dot(h_out,w2)+b2
            h1_out=1/(1+np.exp(-h1))
            y=np.dot(h1_out,w3)+b3
            y_out=1/(1+np.exp(-y))
            if np.argmax(y_out)==np.argmax(pre_output):
                result.append(1)
            else:
                result.append(0)
        pre_result=np.sum(result)/len(result)
        accuracy.append(pre_result)
print(max(accuracy))