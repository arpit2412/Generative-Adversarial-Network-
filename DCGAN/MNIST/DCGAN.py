#!/usr/bin/env python
# coding: utf-8

# In[1]:


##Reference 1: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py
##Reference 2: https://github.com/lyeoni/pytorch-mnist-GAN/blob/master/pytorch-mnist-GAN.ipynb
import argparse
import os
import numpy as np
import math
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

os.makedirs("images", exist_ok=True)
cuda = True if torch.cuda.is_available() else False


# os.makedirs("data/cifar", exist_ok=True)
# transform=transforms.Compose([
#                                transforms.Resize(64),
#                                transforms.ToTensor(),
#                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                            ])
# dataloader = torch.utils.data.DataLoader(
#     datasets.CIFAR10(
#         "data/cifar",
#         train=True,
#         download=True,
#         transform=transform
#     ),
#     batch_size=128,
#     shuffle=True,num_workers=2
# )

# In[30]:


# Configure data loader
os.makedirs("data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=64,
    shuffle=True,
)


# In[9]:


#image size
img_size = 32


# In[17]:


class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        #
        self.init_size = img_size // 4
        #Layer 1
        self.l1 = nn.Sequential(nn.Linear(100, 128 * self.init_size ** 2))
        #Layer 2
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


# In[21]:


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(1, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


# In[22]:


#BCE Loss: Creates a criterion that measures the Binary Cross Entropy between the target and the output:
adversarial_loss = nn.BCELoss()


# In[23]:


# Initialize generator and discriminator
gen = Generator()
dis = Discriminator()


# In[24]:


#model description:Generator
gen


# In[25]:


#model description:Discriminator
dis


# In[26]:


if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()


# In[27]:


#initialize the model 
#normal_ (weight, mean, standard deviation)
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# In[29]:


gen.apply(weights_init_normal)
dis.apply(weights_init_normal)


# In[31]:


lr = 0.0002
#Adam Optimizer
gen_optimizer = optim.Adam(gen.parameters(), lr = lr,betas=(0.5,0.999))
disc_optimizer = optim.Adam(dis.parameters(), lr = lr,betas=(0.5,0.999))


# In[32]:


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# In[36]:


def Gen_train(imgs):
    #clear all gradients i.e. w and b
    gen.zero_grad()
    z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], 100))))

    # Adversarial ground truths
    valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
    fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
    
    #passing noise to generator 
    generator_output = gen(z);
    #now passomg generator output to discriminator: will detect if the generated if fake or real
    disc_output = dis(generator_output)
    #calculating loss 
    loss = adversarial_loss(disc_output,valid)
    #gradient backprop 
    loss.backward()
    #optimize ONLY G's parameters
    gen_optimizer.step()
    return loss.data.item()


# In[37]:


def Disc_train(imgs):
    #clear all gradients i.e. w and b
    dis.zero_grad()
    real_imgs = Variable(imgs.type(Tensor))
    # Adversarial ground truths
    valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
    fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
    #real_loss
    disc_output =  dis(real_imgs)
    real_loss = adversarial_loss(disc_output, valid)
    #fake data
    z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], 100))))
    gen_imgs = gen(z)
    output = dis(gen_imgs.detach())
    fake_loss = adversarial_loss(output, fake)
    fake_score = output                                    
    # gradient backprop & optimize ONLY D's parameters
    disc_loss = (real_loss + fake_loss)/2
    disc_loss.backward()
    disc_optimizer.step()
    return  disc_loss.data.item()


# In[ ]:


n_epoch = 200
for epoch in range(n_epoch):
    D_losses, G_losses = [], []
    for i, (imgs, _) in enumerate(dataloader):
        D_losses.append(Disc_train(imgs))
        G_losses.append(Gen_train(imgs))
        print(imgs.shape[0])
        print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch), n_epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses)))) 


# In[ ]:


with torch.no_grad():
    z = Variable(Tensor(np.random.normal(0, 1, (64, 100))))
    generated = gen(z)
    save_image(generated.view(generated.size(0), 1, 32, 32), 'images/sample_' + '.png')

