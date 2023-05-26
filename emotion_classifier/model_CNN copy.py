import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from dataload_copy import FaceDataset
#对于mac 只需要将\\ 改为/
# 首先 深度学习在gpu中运行 首先就是要模型（model）和损失函数(loss_function)和数据(data)放到gpu中运行 .cuda()
# 在我们重写我们的数据加载类的时候首先需要将数据放到cuda中然后再返回
# 在验证集和训练集中 我们 for循环每一个peach 都需要将其中的数据放到gpu中 (好像不需要这样)只要在 我们的数据加载类中将数据放入到gpu中每次加载数据的时候就都没有问题了
#创建默认的CPU设备.
# device = torch.device("cpu")
#如果GPU设备可用，将默认设备改为GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 参数初始化
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    # 字符串查找find，找不到返回-1，不等-1即字符串中含有该字符
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.04)


# 验证模型在验证集上的正确率
def validate(model, dataset, batch_size):
    val_loader = data.DataLoader(dataset, batch_size)
    total = len(dataset)
    right = 0 
    # loss_function = nn.CrossEntropyLoss()
    # 防止梯度爆炸
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model.forward(images)
            right = right + (outputs.argmax(1)==labels).sum()  # 计数
    acc = right.item()/total
    return acc

class FaceCNN(nn.Module):
    # 初始化网络结构
    def __init__(self):
        super(FaceCNN, self).__init__()

        # 第一次卷积、池化
        self.conv1 = nn.Sequential(
            # 输入通道数in_channels，输出通道数(即卷积核的通道数)out_channels，卷积核大小kernel_size，步长stride，对称填0行列数padding
            # input:(bitch_size, 1, 48, 48), output:(bitch_size, 64, 48, 48), (48-3+2*1)/1+1 = 48
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1), # 卷积层
            nn.BatchNorm2d(num_features=64), # 归一化
            # nn.RReLU(inplace=True), # 激活函数
            nn.Sigmoid(), # 激活函数
            # output(bitch_size, 64, 24, 24)
            nn.MaxPool2d(kernel_size=2, stride=2), # 最大值池化
        )

        # 第二次卷积、池化
        self.conv2 = nn.Sequential(
            # input:(bitch_size, 64, 24, 24), output:(bitch_size, 128, 24, 24), (24-3+2*1)/1+1 = 24
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.RReLU(inplace=True),
            # output:(bitch_size, 128, 12 ,12)
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 第三次卷积、池化
        self.conv3 = nn.Sequential(
            # input:(bitch_size, 128, 12, 12), output:(bitch_size, 256, 12, 12), (12-3+2*1)/1+1 = 12
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.RReLU(inplace=True),
            # output:(bitch_size, 256, 6 ,6)
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 参数初始化
        self.conv1.apply(gaussian_weights_init)
        self.conv2.apply(gaussian_weights_init)
        self.conv3.apply(gaussian_weights_init)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=256*6*6, out_features=4096),
            nn.RReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=1024),
            nn.RReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=256),
            nn.RReLU(inplace=True),
            nn.Linear(in_features=256, out_features=7),
        )

    # 前向传播
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # 数据扁平化
        x = x.view(x.shape[0], -1)
        y = self.fc(x)
        return y    
    
def train(train_dataset, val_dataset, batch_size, epochs, learning_rate, wt_decay):
    #存储损失率样例 以列表的形式存储
    train_loss = []
    train_acc = []
    acc_vall = []
    # 载入数据并分割batch
    train_loader = data.DataLoader(train_dataset, batch_size)
    # 构建模型
    model = FaceCNN()
    checkpoint_save_path = r'Z:\data\model'
    if os.path.exists(r'Z:\data\model\final1.pth'):
        print('-------------load the model-----------------')
        # model.load_state_dict(torch.load(r'Z:\torch test\data\finnal\model\10.pth'))
        model = torch.load(checkpoint_save_path+'/final.pth')
        # model.eval()    # 模型推理时设置
   #如果模型之前训练过，就加载之前的模型继续训练
    model.to(device)
    # 损失函数
    loss_function = nn.CrossEntropyLoss()
    loss_function.to(device)
    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=wt_decay)
    # 学习率衰减
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    # 逐轮训练
    for epoch in range(epochs):
        # 记录损失值
        loss_rate = 0
        # scheduler.step() # 学习率衰减
        model.train()# 模型训练
        for images, emotion in train_loader:
            # 梯度清零 
            # if epoch % 2 ==0:
            optimizer.zero_grad()
            images=images.to(device)
            emotion=emotion.to(device)
            # 前向传播
            output = model.forward(images)
            # 误差计算
            loss_rate = loss_function(output, emotion)
            # 误差的反向传播
            loss_rate.backward()
            # 更新参数
            optimizer.step()

        # 打印每轮的损失

        print('After {} epochs , the loss_rate is : '.format(epoch+1), loss_rate.item())
        train_loss.append(loss_rate.item())
        if epoch % 5 == 0:
            model.eval() # 模型评估
            acc_train = validate(model, train_dataset, batch_size)
            acc_val = validate(model, val_dataset, batch_size)
            print('After {} epochs , the acc_train is : '.format(epoch+1), acc_train)
            print('After {} epochs , the acc_val is : '.format(epoch+1), acc_val)
            train_acc.append(acc_train) 
            acc_vall.append(acc_val)
        model_path = r'Z:\data\model'
        # model_path = '/Users/lanyiwei/data/CK+48'
        if epoch % 10 == 0:
            path = model_path+'/'+ str(epoch) +'.pth'
            torch.save(model,path)
    # with open("r'Z:\torch test\data\finnal\model'\train_loss.txt'", 'w') as train_loss:
    #     train_los.write(str(train_loss))
    # with open("r'Z:\torch test\data\finnal\model'\train_ac.txt'", 'w') as train_acc:
    #     train_ac.write(str(train_acc))
    # with open("r'Z:\torch test\data\finnal\model'\acc_vall.txt'", 'w') as acc_vall:
    #     acc_vall.write(str(acc_vall))
    path1 = r'Z:\data\savedata'
    # path1 = '/Users/lanyiwei/data/savedata'
    np.savetxt(path1+'/train_loss.txt', train_loss, fmt = '%f', delimiter = ',')
    np.savetxt(path1+'/train_acc.txt', train_acc, fmt = '%f', delimiter = ',')
    np.savetxt(path1+'/acc_vall.txt', acc_vall, fmt = '%f', delimiter = ',')
    return model

def main():
    # 数据集实例化(创建数据集)
    train_set = r'Z:\data\CK+48'
    verify_set = r'Z:\data\CK+48'
    # train_set = '/Users/lanyiwei/data/CK+48'
    # verify_set = '/Users/lanyiwei/data/CK+48'
    model_path = r'Z:\data\model'
    # model_path='/Users/lanyiwei/data/model'

    train_dataset = FaceDataset(root= train_set)
    val_dataset = FaceDataset(root =verify_set)
    # 超参数可自行指定
    model = train(train_dataset, val_dataset, batch_size=128, epochs=10, learning_rate=0.01, wt_decay=0)
    # 保存模型
    path = model_path+'/'+ 'final' +'.pth'
    torch.save(model,path)
    # model 是保存模型 model.state_dict() 是保存数据

if __name__ == '__main__':
    main()