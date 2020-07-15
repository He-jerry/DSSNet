#! /usr/bin/env python

import argparse
import os
import numpy as np
import math
import torch
import torch.nn as nn
import torchvision
from dataset import ImageDataset
from tqdm import tqdm
from torch.nn import init
from PIL import Image
from net import network
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn.functional as F
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1 and classname.find('SplAtConv') == -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)
criterion_bce=torch.nn.BCEWithLogitsLoss()

gen = network()
gen.apply(weights_init_kaiming)
writer = SummaryWriter(log_dir="/public/zebanghe2/derain/reimplement/DSSNet/log/", comment="DSS")

gen = gen.cuda()
criterion_bce.cuda()
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
optimizer_G = torch.optim.Adam(gen.parameters(), lr=0.0005, betas=(0.5, 0.999))
#optimizer_T = torch.optim.Adam(netparam, lr=0.00005, betas=(0.5, 0.999))
# Configure dataloaders


trainloader = DataLoader(
    ImageDataset(transforms_=None),
    batch_size=4,
    shuffle=False,drop_last=True
)
print("data length:",len(trainloader))
Tensor = torch.cuda.FloatTensor
eopchnum=50
print("start training")
totalloss_g1=0
totalloss_g2=0
totalloss_g3=0
totalloss_g4=0
for epoch in range(1, eopchnum+1):
  print("epoch:",epoch)
  iteration=0
  gen.train()
  #train
  train_iterator = tqdm(trainloader, total=len(trainloader))
  for total in train_iterator:
    
    iteration=iteration+1
    optimizer_G.zero_grad()
    # Model inputs
    real_img = total["img"]
    real_mask = total["mask"]
    real_img=real_img.cuda()
    real_mask=real_mask.cuda()
    real_img=Variable(real_img,requires_grad=False)
    real_mask=Variable(real_mask,requires_grad=False)
    out4,out3,out2,out1=gen(real_img)
    lossg1=criterion_bce(out1,real_mask)
    lossg2=criterion_bce(out2,real_mask)
    lossg3=criterion_bce(out3,real_mask)
    lossg4=criterion_bce(out4,real_mask)
    lossg=lossg1+lossg2+lossg3+lossg4
    lossg.backward()
    
    del out1,out2,out3,out4
     
    optimizer_G.step()
    if iteration % 5000 == 0:	
      writer.add_scalar('lossg1', lossg1, iteration)	
      writer.add_scalar('lossg2', lossg2, iteration)
      writer.add_scalar('lossg3', lossg3, iteration)	
      writer.add_scalar('lossg4', lossg4, iteration)
    
    #print("batch:%3d,iteration:%3d,loss_g1:%3f,loss_g2:%3f"%(epoch,iteration,lossg1.item(),lossg2.item()))
    train_iterator.set_description("batch:%3d,iteration:%3d,loss_g1:%3f,loss_g2:%3f,loss_g3:%3f,loss_g4:%3f,loss_total:%3f"%(epoch+1,iteration,lossg1.item(),lossg2.item(),lossg3.item(),lossg4.item(),lossg.item()))
    del lossg1,lossg2,lossg3,lossg4,lossg
  if(epoch%5==0):
    torch.save(gen,"/public/zebanghe2/derain/reimplement/residualse/modeltest/ressenet_t1_%s.pth"%epoch)



    