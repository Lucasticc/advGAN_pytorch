import torch.nn as nn
import torch.nn.functional as F


# Target Model definition 目标网络定义
# class MNIST_target_net(nn.Module):
#     def __init__(self):
#         super(MNIST_target_net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
#         self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
#         self.conv4 = nn.Conv2d(64, 64, kernel_size=3)

#         self.fc1 = nn.Linear(64*4*4, 200)
#         self.fc2 = nn.Linear(200, 200)
#         self.logits = nn.Linear(200, 10)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2)
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))
#         x = F.max_pool2d(x, 2)
#         x = x.view(-1, 64*4*4)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, 0.5)
#         x = F.relu(self.fc2(x))
#         x = self.logits(x)
#         return x
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    # 字符串查找find，找不到返回-1，不等-1即字符串中含有该字符
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.04)
class MNIST_target_net(nn.Module):
    def __init__(self):
        super(MNIST_target_net, self).__init__()

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


#辨别器 网络定义
# class Discriminator(nn.Module):
#     def __init__(self, image_nc):
#         super(Discriminator, self).__init__()
#         # MNIST: 1*28*28 conv2d 二维卷机网络输入 和输出
#         model = [
#             nn.Conv2d(image_nc, 8, kernel_size=4, stride=2, padding=0, bias=True),
#             nn.LeakyReLU(0.2),
#             # 8*13*13
#             nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=0, bias=True),
#             nn.BatchNorm2d(16),
#             nn.LeakyReLU(0.2),
#             # 16*5*5
#             nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0, bias=True),
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(32, 1, 1),
#             nn.Sigmoid()
#             # 32*1*1
#         ]   
#         self.model = nn.Sequential(*model)

#     def forward(self, x):
#         output = self.model(x).squeeze()
#         return output

# #生成器 网络
# class Generator(nn.Module):
#     def __init__(self,
#                  gen_input_nc,
#                  image_nc,
#                  ):
#         super(Generator, self).__init__()

#         encoder_lis = [
#             # MNIST:1*28*28
#             nn.Conv2d(gen_input_nc, 8, kernel_size=3, stride=1, padding=0, bias=True),
#             nn.InstanceNorm2d(8),
#             nn.ReLU(),
#             # 8*26*26
#             nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0, bias=True),
#             nn.InstanceNorm2d(16),
#             nn.ReLU(),
#             # 16*12*12
#             nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0, bias=True),
#             nn.InstanceNorm2d(32),
#             nn.ReLU(),
#             # 32*5*5
#         ]

#         bottle_neck_lis = [ResnetBlock(32),
#                        ResnetBlock(32),
#                        ResnetBlock(32),
#                        ResnetBlock(32),]

#         decoder_lis = [
#             nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0, bias=False),
#             nn.InstanceNorm2d(16),
#             nn.ReLU(),
#             # state size. 16 x 11 x 11
#             nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=0, bias=False),
#             nn.InstanceNorm2d(8),
#             nn.ReLU(),
#             # state size. 8 x 23 x 23
#             nn.ConvTranspose2d(8, image_nc, kernel_size=6, stride=1, padding=0, bias=False),
#             nn.Tanh()
#             # state size. image_nc x 28 x 28
#         ]

#         self.encoder = nn.Sequential(*encoder_lis)
#         self.bottle_neck = nn.Sequential(*bottle_neck_lis)
#         self.decoder = nn.Sequential(*decoder_lis)

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.bottle_neck(x)
#         x = self.decoder(x)
#         return x
class Generator(nn.Module):
    def __init__(self, gen_input_nc, image_nc):
        super(Generator, self).__init__()

        # 编码器部分
        encoder_lis = [
            nn.Conv2d(gen_input_nc, 8, kernel_size=3, stride=1, padding=0, bias=True), # 输入通道为gen_input_nc，输出通道为8的卷积层
            nn.InstanceNorm2d(8), # 实例标准化
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
        ]

        # 瓶颈部分
        bottle_neck_lis = [
            ResnetBlock(32),
            ResnetBlock(32),
            ResnetBlock(32),
            ResnetBlock(32),
        ]

        # 解码器部分
        decoder_lis = [
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0, bias=False), # 反卷积层，上采样
            nn.InstanceNorm2d(16), # 实例标准化
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, image_nc, kernel_size=12, stride=2, padding=0, bias=False), # 最后一个卷积层输出通道为image_nc，内核大小为12
            nn.Tanh() # Tanh激活函数
        ]

        self.encoder = nn.Sequential(*encoder_lis) # 编码器
        self.bottle_neck = nn.Sequential(*bottle_neck_lis) # 瓶颈
        self.decoder = nn.Sequential(*decoder_lis) # 解码器

    def forward(self, x):
        x = self.encoder(x) # 编码
        x = self.bottle_neck(x) # 瓶颈
        x = self.decoder(x) # 解码
        return x


# Define a resnet block
# modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out