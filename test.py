import argparse
import os
import numpy
from skimage import io
import numpy as np
import math
import itertools
import sys
import torch
import torch.nn as nn
import torchvision
import os

import cv2
import torchvision.transforms as transforms
from torchvision.utils import save_image

from dataset import ImageDataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
def tensor2im(input_image, imtype=np.uint8):
    mean = [0.5,0.5,0.5] 
    std = [0.5,0.5,0.5]  
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        for i in range(len(mean)):
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255
        #print(image_numpy)
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def save_img(im, path,h,w):
    im_grid = im
    im_numpy = tensor2im(im_grid) 
    im_array = Image.fromarray(im_numpy)
    im_array=im_array.resize((h,w))
    im_array.save(path)

with torch.no_grad():
  #net=generator()
  net=torch.load("/public/zebanghe2/derain/reimplement/residualse/modeltest/ressenet_t1_11150.pth")
net.eval()
net.cuda()

def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)

	dn = (d-mi)/(ma-mi)

	return dn
 
def save_output(image_name,pred,d_dir):

	predict = pred
	predict = predict.squeeze()
	predict_np = predict.cpu().data.numpy()

	im = Image.fromarray(predict_np*255).convert('RGB')
	img_name = image_name.split("/")[-1]
	image = io.imread(image_name)
	imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

	pb_np = np.array(imo)

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]

	imo.save(d_dir+imidx+'.png')


g = os.walk("/public/zebanghe2/derain/reimplement/residualse/test/mix/")  

for path,dir_list,file_list in g:  
    for file_name in file_list:
      tests=Image.open(path+'/'+file_name)
      print(file_name)
      h,w=tests.size
      tests=transform(tests)
      tests=tests.unsqueeze(0)
      tests=tests.cuda()
      with torch.no_grad():
        out1,out2,out3,out4=net(tests)

      #mask1=faketrans.cpu().detach().numpy()[0,:,:,:]
      #mask1=tensor2im(outrf.cpu()[0,:,:,:])
      #save_img(mask1,"/public/zebanghe2/joint/output/"+file_name.split('.')[0]+'_rf.jpg',h,w)
      #print(outmap)
      mask1=tensor2im(out1.cpu()[0,:,:,:])
      save_img(mask1,"/public/zebanghe2/derain/reimplement/DSSNet/result/"+file_name.split('.')[0]+'_sod1.png',h,w)
      mask2=tensor2im(out2.cpu()[0,:,:,:])
      save_img(mask2,"/public/zebanghe2/derain/reimplement/DSSNet/result/"+file_name.split('.')[0]+'_sod2.png',h,w)
      mask2=tensor2im(out3.cpu()[0,:,:,:])
      save_img(mask2,"/public/zebanghe2/derain/reimplement/DSSNet/result/"+file_name.split('.')[0]+'_sod3.png',h,w)
      mask2=tensor2im(out4.cpu()[0,:,:,:])
      save_img(mask2,"/public/zebanghe2/derain/reimplement/DSSNet/result/"+file_name.split('.')[0]+'_sod4.png',h,w)
      #mask1=tensor2im(outmap.cpu()[0,:,:,:])
      #save_img(mask1,"/public/zebanghe2/joint/output/"+file_name.split('.')[0]+'_map.jpg',h,w)
      #mask1=denormalize(faketrans.cpu().data[1])
      #mask1=denormalize(faketrans.cpu().data[2])
      #sodmask=denormalize(sod.cpu().data[0])
      #sodmask=normPRED(sod)
      #sodmask=cv2.resize(sodmask,(h,w))
      #mask1=cv2.resize(mask1,(h,w))
      #cv2.imwrite("/public/zebanghe2/BASGan/output1/"+file_name.split('.')[0]+'_trans.jpg',mask1)
      #cv2.imwrite("/public/zebanghe2/BASGan/output3/"+file_name.split('.')[0]+'_map.png',sodmask)
      #sodmask=outmap.cpu().detach().numpy()[0,0,:,:]
      #smean=np.mean(sodmask)
      #sodmask[sodmask>smean]=255
      #sodmask[sodmask<smean]=0
      #sodmask=cv2.resize(sodmask,(h,w))
      #cv2.imwrite("/public/zebanghe2/derain/reimplement/residualse/result/"+file_name.split('.')[0]+'_map.png',sodmask)
      
