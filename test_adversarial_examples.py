import torch
import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.utils as save
from torch.utils.data import DataLoader
import models
from advGAN import AdvGAN_Attack
from models import MNIST_target_net
from dataload import FaceDataset
from modelcnn import FaceCNN
from models import Discriminator

use_cuda=True
image_nc=1
batch_size = 128

gen_input_nc = image_nc

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# load the pretrained model
# # pretrained_model = "./MNIST_target_model.pth"
# pretrained_model = "/Users/lanyiwei/data/advmodel/20.pth"
# target_model = MNIST_target_net().to(device)
# target_model.load_state_dict(torch.load(pretrained_model))
# target_model.eval()
def main():
    #原网络地址
    pretrained_model = "/Users/lanyiwei/data/model/F.pth"
        # pretrained_model = "./MNIST_target_model.pth"
        #原网络的定义 指向models
    targeted_model = MNIST_target_net().to(device)
        #载入模型
    # targeted_model.load_state_dict(torch.load(pretrained_model))
    targeted_model=torch.load(pretrained_model)
        #评估模式
    targeted_model.eval()

    # load the generator of adversarial examples
    pretrained_generator_path = '/Users/lanyiwei/data/advmodel/netG_epoch_2la90.pth'
    pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
    pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
    pretrained_G.eval()

    # test adversarial examples in MNIST training dataset
    # mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
    # train_dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    train_set = '/Users/lanyiwei/data/verify_set'
    train_dataset = FaceDataset(root= train_set)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    num_correct = 0
    for i, data in enumerate(train_dataloader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        perturbation = pretrained_G(test_img)
        perturbation = torch.clamp(perturbation, -0.3, 0.3)
        adv_img = perturbation + test_img
        path1 = '/Users/lanyiwei/data/image'+'/'+'l'+'.jpg'
        path2 = '/Users/lanyiwei/data/image'+'/'+'a'+'.jpg'
        if i == 9:
            save.save_image(test_img,path1)
            save.save_image(adv_img,path2)
            print('save image')
        adv_img = torch.clamp(adv_img, 0, 1)
        pred_lab = torch.argmax(targeted_model(adv_img),1)
        num_correct += torch.sum(pred_lab==test_label,0)

    print('MNIST training dataset:')
    print('num_correct: ', num_correct.item())
    print('accuracy of adv imgs in training set: %f\n'%(num_correct.item()/len(train_dataset)))

    # # test adversarial examples in MNIST testing dataset
    # mnist_dataset_test = torchvision.datasets.MNIST('./dataset', train=False, transform=transforms.ToTensor(), download=True)
    # test_dataloader = DataLoader(mnist_dataset_test, batch_size=batch_size, shuffle=False, num_workers=1)
    # num_correct = 0
    # for i, data in enumerate(test_dataloader, 0):
    #     test_img, test_label = data
    #     test_img, test_label = test_img.to(device), test_label.to(device)
    #     perturbation = pretrained_G(test_img)
    #     perturbation = torch.clamp(perturbation, -0.3, 0.3)
    #     adv_img = perturbation + test_img
    #     adv_img = torch.clamp(adv_img, 0, 1)
    #     pred_lab = torch.argmax(target_model(adv_img),1)
    #     num_correct += torch.sum(pred_lab==test_label,0)

    # print('num_correct: ', num_correct.item())
    # print('accuracy of adv imgs in testing set: %f\n'%(num_correct.item()/len(mnist_dataset_test)))
if (__name__ == '__main__'):
    main()