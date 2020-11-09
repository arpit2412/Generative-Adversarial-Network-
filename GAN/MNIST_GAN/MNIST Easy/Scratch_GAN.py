#!/usr/bin/env python
# coding: utf-8

# In[143]:


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


# In[144]:


#Batch size 100
os.makedirs("data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=100,
    shuffle=True,
)


# In[145]:


#Generator 
class Generator(nn.Module):
    def __init__(self,input_dim,out_dim):
        super(Generator,self).__init__()
        #input_dim = 100
        ##out_dim = 784
        self.layer1 = nn.Sequential(nn.Linear(input_dim, 256),nn.LeakyReLU(0.2))
        self.layer2 = nn.Sequential(nn.Linear(256, 512),nn.LeakyReLU(0.2))
        self.layer3 = nn.Sequential(nn.Linear(512, 1024),nn.LeakyReLU(0.2))
        self.layer4 = nn.Linear(1024, out_dim)
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        y = nn.Tanh()
        return y(self.layer4(x))     


# In[146]:


def images_to_vectors(images):
    return images.view(images.size(0), 784)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)


# In[147]:


#Discriminator
#Leaky Relu: LeakyReLU(x)=max(0,x)+negative_slope∗min(0,x)
class Discriminator(nn.Module):
    def __init__(self,in_dim):
        super(Discriminator,self).__init__()
        #in_dim = 784
        self.layer1 = nn.Sequential(nn.Linear(in_dim, 1024),nn.LeakyReLU(0.2),nn.Dropout(0.3))
        self.layer2 = nn.Sequential(nn.Linear(1024, 512),nn.LeakyReLU(0.2),nn.Dropout(0.3))
        self.layer3 = nn.Sequential(nn.Linear(512, 256),nn.LeakyReLU(0.2),nn.Dropout(0.3))
        self.layer4 = nn.Linear(256, 1)

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        y = nn.Sigmoid()
        return y(self.layer4(x))    


# In[148]:


#BCE Loss: Creates a criterion that measures the Binary Cross Entropy between the target and the output:
adversarial_loss = nn.BCELoss()


# In[149]:


# Initialize generator and discriminator
gen = Generator(100,784)
dis = Discriminator(784)


# In[150]:


#model description:Generator
gen


# In[151]:


#model description:Discriminator
dis


# In[152]:


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# In[153]:


lr = 0.0002
#Adam Optimizer
gen_optimizer = optim.Adam(gen.parameters(), lr = lr)
disc_optimizer = optim.Adam(dis.parameters(), lr = lr)


# In[154]:


def Gen_train(x):
    #clear all gradients i.e. w and b
    gen.zero_grad()
    z = Variable(torch.randn(100, 100)) #Gaussian Distribution 
    y = Variable(torch.ones(100, 1))   #output 1 initially all 1 as output 
    #passing noise to generator 
    generator_output = gen(z);
    #now passomg generator output to discriminator: will detect if the generated if fake or real
    disc_output = dis(generator_output)
    #calculating loss 
    loss = adversarial_loss(disc_output,y)
    #gradient backprop 
    loss.backward()
    #optimize ONLY G's parameters
    gen_optimizer.step()
    return loss.data.item()


# In[157]:


def Disc_train(x):
    #real_imgs = Variable(x)  #X
    #valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False) #Y
    
    x_real, y_real = x.view(-1, 784), torch.ones(100, 1)
    x_real, y_real = Variable(x_real), Variable(y_real)
    
   
    #clear all gradients i.e. w and b
    dis.zero_grad()
    #real_loss
    disc_output =  dis(x_real)
    real_loss = adversarial_loss(disc_output, y_real)
    #fake data
    z = Variable(torch.randn(100, 100))
    x_fake, y_fake = gen(z), Variable(torch.zeros(100, 1))                                
    output = dis(x_fake)
    fake_loss = adversarial_loss(output, y_fake)
    fake_score = output                                    
    # gradient backprop & optimize ONLY D's parameters
    disc_loss = real_loss + fake_loss
    disc_loss.backward()
    disc_optimizer.step()
    return  disc_loss.data.item()


# In[ ]:


n_epoch = 200
for epoch in range(1, n_epoch+1):
    D_losses, G_losses = [], []
    for i, (imgs, _) in enumerate(dataloader):
        D_losses.append(Disc_train(imgs))
        G_losses.append(Gen_train(imgs))
        
    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch), n_epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))       


# In[ ]:


with torch.no_grad():
    test_z = Variable(torch.randn(100, 100))
    generated = gen(test_z)
    save_image(generated.view(generated.size(0), 1, 28, 28), './samples/sample_' + '.png')

