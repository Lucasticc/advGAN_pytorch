import torch.utils.data as data
import torch
import os
import torch.nn as nn
import numpy as np
# from dataload import FaceDataset
from emotion_classifier.dataload_copy import FaceDataset
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")



class FaceCNN(nn.Module):
    # 前向传播
    def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            # 数据扁平化
            x = x.view(x.shape[0], -1)
            y = self.fc(x)
            return y    
def validate(model, dataset, batch_size):
    val_loader = data.DataLoader(dataset, batch_size)
    total = len(dataset)
    print(total)
    right = 0 
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels= labels.to(device)
            outputs = model.forward(images)
            right = right + (outputs.argmax(1)==labels).sum()  # 计数
    print(right.item())
    acc = right.item()/total
    return acc
def train(val_dataset, batch_size):
    acc_vall = []
    checkpoint_save_path = '/Users/lanyiwei/data'
    path=r'Z:\data\model\95.pth'
    if os.path.exists('/Users/lanyiwei/data/model/0.pth'):
        print('-------------load the model-----------------')
        model = torch.load(checkpoint_save_path+'/140.pth')
        model.to(device)
        model.eval() # 模型评估
        acc_val = validate(model, val_dataset, batch_size)
        print('After {} epochs , the acc_val is : '.format(1), acc_val)
        # acc_vall.append(acc_val)
    model = torch.load(path)
    model.eval() # 模型评估
    acc_val = validate(model, val_dataset, batch_size)
    print('After {} epochs , the acc_val is : '.format(1), acc_val)
            
    # path1 = '/Users/lanyiwei/data/savedata'
    # np.savetxt(path1+'/single_test.txt', acc_vall, fmt = '%f', delimiter = ',')

def main():
    # verify_set = '/Users/lanyiwei/data/verify_set'
    verify_set = '/Users/lanyiwei/data/CK+48'
    verify_set=r'Z:\data\CK+48'
    val_dataset = FaceDataset(root =verify_set)
    train(val_dataset, batch_size=128)

if __name__ == '__main__':
    main()