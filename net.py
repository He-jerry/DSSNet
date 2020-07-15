from resnest.torch import resnest50
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
class fen(nn.Module):
  def __init__(self):
    super(fen,self).__init__()
    net=resnest50(pretrained=True)
    netlist=list(net.children())
    self.fe1=nn.Sequential(*netlist[0:4])#64
    self.fe2=nn.Sequential(*netlist[4])#256
    self.fe3=nn.Sequential(*netlist[5])#512
    self.fe4=nn.Sequential(*netlist[6])#1024
    self.fe5=nn.Sequential(*netlist[7])#2048
  def forward(self,x):
    fe1=self.fe1(x)
    fe2=self.fe2(fe1)
    fe3=self.fe3(fe2)
    fe4=self.fe4(fe3)
    fe5=self.fe5(fe4)
    return fe1,fe2,fe3,fe4,fe5
    
class network(nn.Module):
  def __init__(self):
    super(network,self).__init__()
    self.fen=fen()
    for p in self.parameters():
       p.requires_grad = False
    self.sig=nn.Sigmoid()
    self.relu=nn.ReLU(inplace=True)
    
    #branch1
    self.conv11=nn.Conv2d(256+1,128,kernel_size=3,stride=1,padding=1)
    self.bn11=nn.BatchNorm2d(128)
    self.conv12=nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1)
    self.bn12=nn.BatchNorm2d(64)
    self.conv13=nn.Conv2d(64,1,kernel_size=3,stride=1,padding=1)
    
    #branch2
    self.conv21=nn.Conv2d(512+1,256,kernel_size=3,stride=1,padding=1)
    self.bn21=nn.BatchNorm2d(256)
    self.conv22=nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1)
    self.bn22=nn.BatchNorm2d(128)
    self.conv23=nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1)
    self.bn23=nn.BatchNorm2d(64)
    self.conv24=nn.Conv2d(64,1,kernel_size=3,stride=1,padding=1)
    
    #branch3
    self.conv31=nn.Conv2d(1024+1,512,kernel_size=3,stride=1,padding=1)
    self.bn31=nn.BatchNorm2d(512)
    self.conv32=nn.Conv2d(512,256,kernel_size=3,stride=1,padding=1)
    self.bn32=nn.BatchNorm2d(256)
    self.conv33=nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1)
    self.bn33=nn.BatchNorm2d(128)
    self.conv34=nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1)
    self.bn34=nn.BatchNorm2d(64)
    self.conv35=nn.Conv2d(64,1,kernel_size=3,stride=1,padding=1)
    
    #branch4
    self.conv41=nn.Conv2d(2048,1024,kernel_size=3,stride=1,padding=1)
    self.bn41=nn.BatchNorm2d(1024)
    self.conv42=nn.Conv2d(1024,512,kernel_size=3,stride=1,padding=1)
    self.bn42=nn.BatchNorm2d(512)
    self.conv43=nn.Conv2d(512,256,kernel_size=3,stride=1,padding=1)
    self.bn43=nn.BatchNorm2d(256)
    self.conv44=nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1)
    self.bn44=nn.BatchNorm2d(128)
    self.conv45=nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1)
    self.bn45=nn.BatchNorm2d(64)
    self.conv46=nn.Conv2d(64,1,kernel_size=3,stride=1,padding=1)
  def forward(self,x):
    fe1,fe2,fe3,fe4,fe5=self.fen(x)
    #branch4
    c51=F.relu(self.bn41(self.conv41(fe5)))
    c52=F.relu(self.bn42(self.conv42(c51)))
    c53=F.relu(self.bn43(self.conv43(c52)))
    c54=F.relu(self.bn44(self.conv44(c53)))
    c55=F.relu(self.bn45(self.conv45(c54)))
    c56=F.sigmoid(self.conv46(c55))
    out4=F.interpolate(c56,size=(384,384))
    
    bd4=F.interpolate(c56,size=(fe4.shape[3],fe4.shape[2]))
    bd4=torch.cat([fe4,bd4],1)
    #branch3
    c41=F.relu(self.bn31(self.conv31(bd4)))
    c42=F.relu(self.bn32(self.conv32(c41)))
    c43=F.relu(self.bn33(self.conv33(c42)))
    c44=F.relu(self.bn34(self.conv34(c43)))
    c45=F.sigmoid(self.conv35(c44))
    out3=F.interpolate(c45,size=(384,384))
    
    bd3=F.interpolate(c45,size=(fe3.shape[3],fe3.shape[2]))
    bd3=torch.cat([fe3,bd3],1)
    #branch2
    c31=F.relu(self.bn21(self.conv21(bd3)))
    c32=F.relu(self.bn22(self.conv22(c31)))
    c33=F.relu(self.bn23(self.conv23(c32)))
    c34=F.sigmoid(self.conv24(c33))
    out2=F.interpolate(c34,size=(384,384))
    
    bd2=F.interpolate(c34,size=(fe2.shape[3],fe2.shape[2]))
    bd2=torch.cat([fe2,bd2],1)
    #branch2
    c21=F.relu(self.bn11(self.conv11(bd2)))
    c22=F.relu(self.bn12(self.conv12(c21)))
    c23=F.sigmoid(self.conv13(c22))
    out1=F.interpolate(c34,size=(384,384))
    
    return out4,out3,out2,out1
    
    
    
    
  
    
    
    