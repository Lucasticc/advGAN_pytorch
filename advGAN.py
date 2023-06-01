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
models_path = r'Z:\data\adv_models'


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
        # pretrained_generator_path = '/Users/lanyiwei/data/advmodel/netG_epoch_la60.pth'
        # self.netG.load_state_dict(torch.load(pretrained_generator_path))
        self.netDisc = models.Discriminator(image_nc).to(device)
        pretrained_d_path = '/Users/lanyiwei/data/advmodel/netD_epoch_la60.pth'
        # self.netDisc.load_state_dict(torch.load(pretrained_d_path)) #辨别器网络
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
        # optimize D 生成器
        for i in range(1):
            perturbation = self.netG(x) # torch.size(128,1,28,28) x经过网络生成干扰图像

            # add a clipping trick 指定边界框限制扰动在一定大小
            adv_images = torch.clamp(perturbation, -0.3, 0.3) + x #将输入的张量压缩到一个区间内
            adv_images = torch.clamp(adv_images, self.box_min, self.box_max) #box在main（）函数中规定大小


            #辨别器 梯度清零
            self.optimizer_D.zero_grad()
            pred_real = self.netDisc(x) #辨别器 正常图片通过辨别器
            #平均平方误差损失 mse——loss函数 计算模型预测值与真实值之间误差的损失函数
            loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device)) #辨别器网路loss
            loss_D_real.backward()

            pred_fake = self.netDisc(adv_images.detach()) #detach 返回一个新的tensor，是从当前计算图中分离下来的，但是仍指向原变量的存放位置 不具有梯度
            loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device))
            #具体来说，torch.zeros_like(pred_fake, device=self.device) 生成了一个与 pred_fake 张量相同形状的全零张量，
            # 并且位于与 pred_fake 相同的设备上（通常是 GPU 上）。然后，F.mse_loss 函数计算 pred_fake 和这个全零张量之间的均方误差。
            # 这个均方误差表示生成样本被判别器判断为真实样本的程度，即虚假损失。
            loss_D_fake.backward()
            #loss_D_real旨在拉近吃正样本之后的输出与1的距离  loss_D_fake旨在拉近吃负样本之后与0的距离 负样本也就是对抗样本
            #双向优化
            loss_D_GAN = loss_D_fake + loss_D_real
            self.optimizer_D.step()

        # optimize G 判别器
        for i in range(1):
            self.optimizer_G.zero_grad()

            # cal G's loss in GAN 辨别器的损失函数 
            pred_fake = self.netDisc(adv_images)
            loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device=self.device))
            loss_G_fake.backward(retain_graph=True) #retain_graph = True的意思是保留当前方向传播的计算图，可以做梯度累加 

            # calculate perturbation norm 扰动范数 限制扰动的大小

            # C = 0.9
            C = 0.3
            loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))
            loss_perturb = torch.max(loss_perturb - C, torch.zeros(1, device=self.device))

            # cal adv loss
            logits_model = self.model(adv_images) #传入目标检测网络
            probs_model = F.softmax(logits_model, dim=1) #使用softmax损失函数
            # torch.eye 生成对角线是1其余全为0的矩阵
            onehot_labels = torch.eye(self.model_num_labels, device=self.device)[labels]
            # C&W loss function
            real = torch.sum(onehot_labels * probs_model, dim=1)   #real的损失函数 目标网络的损失函数 
            other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
            zeros = torch.zeros_like(other) # zeros 将该形状的tensor格式所有数值重置为0
            #生成对抗网络的loss  真实的损失函数 减other
            loss_adv = torch.max(real - other, zeros)
            loss_adv = torch.sum(loss_adv)

            # maximize cross_entropy loss
            loss_adv = -F.mse_loss(logits_model, onehot_labels)
            loss_adv = - F.cross_entropy(logits_model, labels)

            adv_lambda = 10
            pert_lambda = 1
            loss_G = adv_lambda * loss_adv + pert_lambda * loss_perturb
            loss_G.backward()
            self.optimizer_G.step()

        return loss_D_GAN.item(), loss_G_fake.item(), loss_perturb.item(), loss_adv.item()
    #loss_D_GAN.item()：判别器对于真实样本和生成样本的判别损失。该损失函数衡量了判别器在正确地将真实样本判别为真实样本和将生成样本判别为生成样本方面的表现。
    # 最小化该损失函数可以让判别器更加准确地区分真实样本和生成样本，从而提高GAN的训练效果。
    # loss_G_fake.item()：生成器对于生成样本的虚假损失。该损失函数衡量了生成器是否能够成功地生成“虚假”样本，即能够欺骗判别器，让其将生成样本误认为是真实样本。
    # 最小化该损失函数可以让生成器生成的样本更接近真实样本，从而提高GAN的训练效果。
    # loss_perturb.item()：生成器对于原始输入的扰动的正则化损失。该损失函数对生成器生成的扰动向量进行正则化，以防止扰动向量过大导致分类错误或无法成功攻击目标神经网络。
    # 通过限制扰动向量的大小，可以确保对抗样本的有效性和稳定性。最小化该损失函数可以使扰动向量尽可能的小，从而保证对抗样本的有效性和稳定性。
    # loss_adv.item()：对抗性攻击的交叉熵损失。该损失函数用于对抗性攻击，衡量了攻击者在生成对抗样本时的效果。
    # 攻击者的目标是使得神经网络在对抗样本上的分类结果与真实标签之间的差距最大化。通过最小化对抗性损失，可以使得生成器生成的对抗样本更加接近真实样本，并能够成功地欺骗目标神经网络。

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
            print("epoch %d:\nloss_D与原图像相似度,判别器损失函数: %.3f,\n loss_G_fake生成器损失函数: %.3f,\
             \nloss_perturb扰动损失函数: %.3f, \nloss_adv: %.3f, \n" %
                  (epoch, loss_D_sum/num_batch, loss_G_fake_sum/num_batch,
                   loss_perturb_sum/num_batch, loss_adv_sum/num_batch))

            # save generator
            if epoch%10==0:
                netG_file_name = models_path + '/netG_epoch_2la' + str(epoch) + '.pth'
                netD_file_name = models_path + '/netD_epoch_2la' + str(epoch) + '.pth'
                torch.save(self.netG.state_dict(), netG_file_name)
                torch.save(self.netDisc.state_dict(), netD_file_name)
        pathtxt = '/Users/lanyiwei/data/savedata'
        pathtxt = r'Z:\data\adv_models'
        np.savetxt(pathtxt+'/loss_D_sums1.txt', loss_D_sums, fmt = '%f', delimiter = ',')
        np.savetxt(pathtxt+'/loss_G_fake_sums1.txt', loss_G_fake_sums, fmt = '%f', delimiter = ',')
        np.savetxt(pathtxt+'/loss_perturb_sum1s.txt', loss_perturb_sums, fmt = '%f', delimiter = ',')
        np.savetxt(pathtxt+'/loss_adv_sums1.txt', loss_adv_sums, fmt = '%f', delimiter = ',')

            

