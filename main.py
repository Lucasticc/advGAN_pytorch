import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from advGAN import AdvGAN_Attack
from models import MNIST_target_net
from modelcnn import FaceCNN

use_cuda=True
image_nc=1
epochs = 60
batch_size = 128
BOX_MIN = 0
BOX_MAX = 1

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

#原网络地址
pretrained_model = "/Users/lanyiwei/data/model/0.pth"
#原网络的定义 指向models
targeted_model = MNIST_target_net().to(device)
#载入模型
# targeted_model.load_state_dict(torch.load(pretrained_model))
targeted_model.load_state_dict(torch.load(pretrained_model))
#评估模式
targeted_model.eval()
model_num_labels = 10

# MNIST train dataset and dataloader declaration 获取数据集 训练模型
mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
advGAN = AdvGAN_Attack(device,
                          targeted_model,
                          model_num_labels,
                          image_nc,
                          BOX_MIN,
                          BOX_MAX)

advGAN.train(dataloader, epochs)
