import torch.nn as nn
import torch
import numpy as np
import models
import torch.nn.functional as F
import torchvision
import os
from models import Generator

#存放 model的路径
# models_path = './models/'
models_path = '/Users/lanyiwei/data/advmodel'


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class AdvGAN_Attack:
    def __init__(self,
                 device,
                 model,
                 model_num_labels,
                 image_nc,
                 box_min,
                 box_max):
        output_nc = image_nc
        self.device = device
        self.model_num_labels = model_num_labels
        self.model = model
        self.input_nc = image_nc
        self.output_nc = output_nc
        self.box_min = box_min
        self.box_max = box_max
        self.gen_input_nc = image_nc
        self.netG = models.Generator(self.gen_input_nc, image_nc).to(device)
        pretrained_generator_path = '/Users/lanyiwei/data/advmodel/netG_epoch_la60.pth'
        self.netG.load_state_dict(torch.load(pretrained_generator_path))
        self.netDisc = models.Discriminator(image_nc).to(device)
        pretrained_d_path = '/Users/lanyiwei/data/advmodel/netD_epoch_la60.pth'
        self.netDisc.load_state_dict(torch.load(pretrained_d_path))
        print('-----load the model-----')

        # initialize all weights
        self.netG.apply(weights_init)
        self.netDisc.apply(weights_init)

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=0.001)
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                            lr=0.001)

        if not os.path.exists(models_path):
            os.makedirs(models_path)

    def train_batch(self, x, labels):
        #生成器 生成干扰图像 x是输入的图片
        # optimize D 辨别器
        for i in range(1):
            perturbation = self.netG(x) # torch.size(128,1,28,28) x经过网络生成干扰图像

            # add a clipping trick
            adv_images = torch.clamp(perturbation, -0.3, 0.3) + x #将输入的张量压缩到一个区间内
            adv_images = torch.clamp(adv_images, self.box_min, self.box_max) #box在main（）函数中规定大小
            # print(adv_images)


            #辨别器 梯度清零
            self.optimizer_D.zero_grad()
            pred_real = self.netDisc(x) #辨别器 正常图片通过辨别器
            #平均平方误差损失 mse——loss函数 计算模型预测值与真实值之间误差的损失函数
            loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device)) #辨别器网路loss
            loss_D_real.backward()

            pred_fake = self.netDisc(adv_images.detach()) #detach 返回一个新的tensor，是从当前计算图中分离下来的，但是仍指向原变量的存放位置 不具有梯度
            loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device))
            loss_D_fake.backward()
            #loss_D_real旨在拉近吃正样本之后的输出与1的距离  oss_D_fake旨在拉近吃负样本之后与0的距离 负样本也就是对抗样本
            loss_D_GAN = loss_D_fake + loss_D_real
            self.optimizer_D.step()

        # optimize G
        for i in range(1):
            self.optimizer_G.zero_grad()

            # cal G's loss in GAN 辨别器的损失函数 
            pred_fake = self.netDisc(adv_images)
            loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device=self.device))
            loss_G_fake.backward(retain_graph=True) #retain_graph = True的意思是保留当前方向传播的计算图，可以做梯度累加 

            # calculate perturbation norm 扰动范数 限制扰动的大小

            C = 0.9
            loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))
            loss_perturb = torch.max(loss_perturb - C, torch.zeros(1, device=self.device))

            # cal adv loss
            logits_model = self.model(adv_images) #传入目标检测网络
            probs_model = F.softmax(logits_model, dim=1) #使用softmax损失函数
            # torch.eye 生成对角线是1其余全为0的矩阵
            onehot_labels = torch.eye(self.model_num_labels, device=self.device)[labels]
            # C&W loss function
            # torch.reshape(probs_model, (896,))
            # add = torch.zeros(4)
            # print(probs_model.shape)
            # probs_model = torch.cat((probs_model, add), 0)
            # print(probs_model.shape, onehot_labels.shape)
            real = torch.sum(onehot_labels * probs_model, dim=1)   #real的损失函数 目标网络的损失函数 
            other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
            zeros = torch.zeros_like(other) # zeros 将该形状的tensor格式所有数值重置为0
            #生成对抗网络的loss  真实的损失函数 减other
            loss_adv = torch.max(real - other, zeros)
            loss_adv = torch.sum(loss_adv)

            # maximize cross_entropy loss
            # loss_adv = -F.mse_loss(logits_model, onehot_labels)
            # loss_adv = - F.cross_entropy(logits_model, labels)

            adv_lambda = 10
            pert_lambda = 1
            loss_G = adv_lambda * loss_adv + pert_lambda * loss_perturb
            loss_G.backward()
            self.optimizer_G.step()

        return loss_D_GAN.item(), loss_G_fake.item(), loss_perturb.item(), loss_adv.item()

    def train(self, train_dataloader, epochs):
        loss_D_sums = []
        loss_G_fake_sums= []
        loss_perturb_sums=[]
        loss_adv_sums=[]
        for epoch in range(1, epochs+1):
            # 更改学习率 以达到更好的效果
            if epoch == 50:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.0001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.0001)
            if epoch == 80:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.00001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.00001)
            loss_D_sum = 0
            loss_G_fake_sum = 0
            loss_perturb_sum = 0
            loss_adv_sum = 0
            for i, data in enumerate(train_dataloader, start=0):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                loss_D_batch, loss_G_fake_batch, loss_perturb_batch, loss_adv_batch = \
                    self.train_batch(images, labels) #从训练集中拿到图片
                loss_D_sum += loss_D_batch
                loss_G_fake_sum += loss_G_fake_batch
                loss_perturb_sum += loss_perturb_batch
                loss_adv_sum += loss_adv_batch

            # print statistics
            num_batch = len(train_dataloader)
            #保存到txt文件
            loss_D_sums.append(loss_D_sum/num_batch)
            loss_G_fake_sums.append(loss_G_fake_sum/num_batch)
            loss_perturb_sums.append(loss_perturb_sum/num_batch)
            loss_adv_sums.append(loss_adv_sum/num_batch)
            print("epoch %d:\nloss_D‘与原图像相似度’: %.3f, loss_G_fake‘辨别器损失函数’: %.3f,\
             \nloss_perturb‘扰动损失函数’: %.3f, loss_adv: %.3f, \n" %
                  (epoch, loss_D_sum/num_batch, loss_G_fake_sum/num_batch,
                   loss_perturb_sum/num_batch, loss_adv_sum/num_batch))

            # save generator
            if epoch%10==0:
                netG_file_name = models_path + '/netG_epoch_2la' + str(epoch) + '.pth'
                netD_file_name = models_path + '/netD_epoch_2la' + str(epoch) + '.pth'
                torch.save(self.netG.state_dict(), netG_file_name)
                torch.save(self.netDisc.state_dict(), netD_file_name)
        pathtxt = '/Users/lanyiwei/data/savedata'
        np.savetxt(pathtxt+'/loss_D_sums1.txt', loss_D_sums, fmt = '%f', delimiter = ',')
        np.savetxt(pathtxt+'/loss_G_fake_sums1.txt', loss_G_fake_sums, fmt = '%f', delimiter = ',')
        np.savetxt(pathtxt+'/loss_perturb_sum1s.txt', loss_perturb_sums, fmt = '%f', delimiter = ',')
        np.savetxt(pathtxt+'/loss_adv_sums1.txt', loss_adv_sums, fmt = '%f', delimiter = ',')

            

