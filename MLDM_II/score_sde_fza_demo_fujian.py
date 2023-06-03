#@title Autoload all modules
#%load_ext autoreload
#%autoreload 2

from pixel import pixel,pixel3,pixel_3
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import io
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import importlib
import os
import functools
import itertools
import torch
from losses import get_optimizer
from models.ema import ExponentialMovingAverage

import torch.nn as nn
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_gan as tfgan
import tqdm
import io
import likelihood
#import controllable_generation_fza
from utils import restore_checkpoint
sns.set(font_scale=2)
sns.set(style="whitegrid")
import cv2
import models
from models import utils as mutils
from models import ncsnv2
from models import ncsnpp
from models import ddpm as ddpm_model
from models import layerspp
from models import layers
from models import normalization
import sampling_fza
import sampling_pc
#import sampling_3noise
#import sampling_3noise_fujian
import sampling_3noise_fujian_mask
from likelihood import get_likelihood_fn
from sde_lib import VESDE, VPSDE, subVPSDE
from sampling_3noise_fujian_mask import (ReverseDiffusionPredictor, 
                      LangevinCorrector, 
                      EulerMaruyamaPredictor, 
                      AncestralSamplingPredictor, 
                      NoneCorrector, 
                      NonePredictor,
                      AnnealedLangevinDynamics)
import datasets
import scipy.io as io
from operator_fza import forward,backward,forward_torch,backward_torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# @title Load the score-based model
sde = 'VESDE' #@param ['VESDE', 'VPSDE', 'subVPSDE'] {"type": "string"}
if sde.lower() == 'vesde':
  #from configs.ve import cifar10_ncsnpp as configs
  #ckpt_filename = "exp/ve/cifar10_ncsnpp/checkpoint_9.pth"
  #from configs.ve import bedroom_ncsnpp_continuous as configs
  #ckpt_filename = "exp_train_bedroom_max380_N1000/checkpoints/checkpoint_50.pth"
  #from configs.ve import bedroom_ncsnpp_continuous as configs
  #ckpt_filename = "exp/ve/bedroom_ncsnpp_continuous/checkpoint_127.pth"
  from configs.ve import church_ncsnpp_continuous as configs
  ckpt_filename = "exp_train_church_6channel_max380_N1000/checkpoints/checkpoint_10.pth" #(9:(20.2,0.5)
  #ckpt_filename = "exp_train_bedroom_6channel_max380_N1000/checkpoints/checkpoint_10.pth"
  #ckpt_filename = "/home/lqg/桌面/score_sde_fza/exp_train10000_church_6channel_max380/checkpoints/checkpoint_30.pth"
  #from configs.ve import church_ncsnpp_continuous as configs
  #ckpt_filename = "exp/ve/church_ncsnpp_continuous/checkpoint_126.pth"
  config = configs.get_config()
  sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
  sampling_eps = 1e-5
elif sde.lower() == 'vpsde':
  from configs.vp import cifar10_ddpmpp_continuous as configs  
  ckpt_filename = "exp/vp/cifar10_ddpmpp_continuous/checkpoint_8.pth"
  config = configs.get_config()
  sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
  sampling_eps = 1e-3
elif sde.lower() == 'subvpsde':
  from configs.subvp import cifar10_ddpmpp_continuous as configs
  ckpt_filename = "exp/subvp/cifar10_ddpmpp_continuous/checkpoint_26.pth"
  config = configs.get_config()
  sde = subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
  sampling_eps = 1e-3

batch_size =  1 #64#@param {"type":"integer"}
config.training.batch_size = batch_size
config.eval.batch_size = batch_size

random_seed = 0 #@param {"type": "integer"}

sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
score_model = mutils.create_model(config)

optimizer = get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(),
                               decay=config.model.ema_rate)
state = dict(step=0, optimizer=optimizer,
             model=score_model, ema=ema)

state = restore_checkpoint(ckpt_filename, state, config.device)

ema.copy_to(score_model.parameters())

#@title Visualization code

def image_grid(x):
  size = config.data.image_size
  channels = config.data.num_channels
  img = x.reshape(-1, size, size, channels)
  w = int(np.sqrt(img.shape[0]))
  img = img.reshape((w, w, size, size, channels)).transpose((0, 2, 1, 3, 4)).reshape((w * size, w * size, channels))
  #img = img.reshape(( size, size, channels*2))
  return img

def show_samples(x):
  x = x.permute(0, 2, 3, 1).detach().cpu().numpy()
  img = image_grid(x)
  plt.figure(figsize=(8,8))
  plt.axis('off')
  plt.imshow(img)
  plt.show()



#@title PC inpainting

predictor = ReverseDiffusionPredictor #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
corrector = LangevinCorrector #@param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}
snr = 0.06 #@param {"type": "number"}
n_steps = 1 #@param {"type": "integer"}
probability_flow = False #@param {"type": "boolean"}
'''
pc_inpainter =  controllable_generation_fza.get_pc_inpainter(sde,
                                                        predictor, corrector,
                                                        inverse_scaler,
                                                        snr=snr,
                                                        n_steps=n_steps,
                                                        probability_flow=probability_flow,
                                                        continuous=config.training.continuous,
                                                        denoise=True)
'''
'''
batch = next(eval_iter)
img = batch['image']._numpy()
print(img)
plt.f 
4gfh igure(figsize=(8,8))
plt.axis('off')
plt.imshow(img[0,:,:,:])
plt.show()
'''

psnr_result=[ ]
ssim_result=[ ]
for j in range(0,1,1):
  print('****************'+'第{}张图'.format(j+1)+'******************')
  img = io.loadmat('./input_output/ori/test_tu/img/church/Img02.mat')['Img']
  '''
  img_ob = io.loadmat('./input_output/ori/test_tu/ob/church_2phase.mat')['ob']
  img_ob1 = img_ob[0:256,0:256]
  img_ob2 = img_ob[0:256,256:512]
  '''
 # img2 = io.loadmat('./input_output/ori/test_tu/house_0.5pi_Img.mat')['Img']
 # img3 = io.loadmat('./input_output/ori/test_tu/house_pi_Img.mat')['Img']
 # img4 = io.loadmat('./input_output/ori/test_tu/house_1.5pi_Img.mat')['Img']

 # img_ob1 = io.loadmat('./input_output/ori/test_tu/ob/bedroom/ob{:0>2d}.mat'.format(j))['ob']       ########################## 
 # img_ob2 = io.loadmat('./input_output/ori/test_tu/ob/bedroom_0.5pi/ob{:0>2d}.mat'.format(j))['ob']
 # img_ob3 = io.loadmat('./input_output/ori/test_tu/1/church_r1=0.25_ob.mat')['ob']
 # img_ob4 = io.loadmat('./input_output/ori/test_tu/1/church_r1=0.25_0.5pi_ob.mat')['ob']

#######################################################################################################
  '''
  img_ob_b=cv2.imread('./input_output/ori/test_tu/MLDM/img/z=50/8_B/Capture_00001.png',-1)
  img_ob_g=cv2.imread('./input_output/ori/test_tu/MLDM/img/z=50/8_G/Capture_00001.png',-1)
  img_ob_r=cv2.imread('./input_output/ori/test_tu/MLDM/img/z=50/8_R/Capture_00001.png',-1)
  '''
  
  img_ob_b=cv2.imread('./input_output/ori/test_tu/MLDM/USAF/3.5 ms/Capture_00001.png',-1)

  img_ob_g=cv2.imread('./input_output/ori/test_tu/MLDM/USAF/3.5 ms/Capture_00001.png',-1)

  img_ob_r=cv2.imread('./input_output/ori/test_tu/MLDM/USAF/3.5 ms/Capture_00001.png',-1)
  
  img_ob_1=np.stack((img_ob_r[:,:],img_ob_g[:,:],img_ob_b[:,:]),axis=2)
 # img_ob1=img_ob_1[100:1380,2725:4005,:]
 # img_ob2=img_ob_1[100:1380,770:2050,:]
  img_ob1=img_ob_1[306:1074,2956:3724,:]
  img_ob2=img_ob_1[296:1064,966:1734,:]
  #img_ob2=img_ob_1[2451:3219,1026:1794,:]
  #img_ob2=img_ob_1[2451:3219,2981:3749,:]
  #img_ob2=img_ob_1[356:1124,2981:3749,:]

  img_ob1=pixel3(img_ob1)
  img_ob2=pixel3(img_ob2)
  
  #print(img_ob1,img_ob1.max(),img_ob2,img_ob2.max())
  #assert 0
  
  #归一化乘系数的参数1
  '''
  img_ob1[:,:,0]=img_ob1[:,:,0]-0.02
  img_ob1[:,:,1]=img_ob1[:,:,1]+0.11
  img_ob1[:,:,2]=img_ob1[:,:,2]+0.13
  img_ob2[:,:,0]=img_ob2[:,:,0]-0.04
  img_ob2[:,:,1]=img_ob2[:,:,1]+0.07
  img_ob2[:,:,2]=img_ob2[:,:,2]+0.11
  '''
  '''
  #归一化乘系数的参数2
  img_ob1[:,:,0]=img_ob1[:,:,0]-0.3
  img_ob1[:,:,1]=img_ob1[:,:,1]-0.12
  img_ob1[:,:,2]=img_ob1[:,:,2]-0.06
  img_ob2[:,:,0]=img_ob2[:,:,0]-0.31
  img_ob2[:,:,1]=img_ob2[:,:,1]-0.13
  img_ob2[:,:,2]=img_ob2[:,:,2]-0.08
  '''
  '''
  #全归一化参数1
  img_ob1[:,:,0]=img_ob1[:,:,0]-0.6
  img_ob1[:,:,1]=img_ob1[:,:,1]
  img_ob1[:,:,2]=img_ob1[:,:,2]+0.2
  img_ob2[:,:,0]=img_ob2[:,:,0]-0.63
  img_ob2[:,:,1]=img_ob2[:,:,1]-0.03
  img_ob2[:,:,2]=img_ob2[:,:,2]+0.16
  '''

  #img_ob1=(img_ob1[:,:,:]-img_ob1.min())/(img_ob1.max()-img_ob1.min())
  #img_ob2=(img_ob2[:,:,:]-img_ob2.min())/(img_ob2.max()-img_ob2.min())

########################################################################################################################

  img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).cuda()     #tensor:(1,3,256,256)
#  img2 = torch.from_numpy(img2).permute(2,0,1).unsqueeze(0).cuda()
#  img3 = torch.from_numpy(img3).permute(2,0,1).unsqueeze(0).cuda()
#  img4 = torch.from_numpy(img4).permute(2,0,1).unsqueeze(0).cuda()

  img_ob1=torch.from_numpy(img_ob1).cuda()                         #######################
  img_ob2=torch.from_numpy(img_ob2).cuda()
 # img_ob3=torch.from_numpy(img_ob3).cuda()
 # img_ob4=torch.from_numpy(img_ob4).cuda()

  '''
  img=cv2.imread('1.jpg',-1)/255
  img=np.stack((img[:,:,2],img[:,:,1],img[:,:,0]),axis=2)
  print(img.shape)
  img=cv2.resize(img,(256,256))
  img=np.expand_dims(img,axis=0)
  '''

 # img = torch.from_numpy(img).permute(0, 3, 1, 2).to(config.device) #1,3,256,256
  #show_samples(img)
  
  dp=0.0038*3
  di=3.4
  z1=500
  r1_1=0.4
  r1_2=0.4
  M=di/z1
  ri_1=(1+M)*r1_1
  ri_2=(1+M)*r1_2
  '''
  # NX,NY=1280,1280
  dp=0.014
  di=3
  z1=300
  r1=0.23
  M=di/z1
  ri_1=(1+M)*r1
  '''
  NX,NY=256,256

  fu_max,fv_max=0.5/dp,0.5/dp
  du,dv=2*fu_max/NX,2*fv_max/NY
  u,v=np.mgrid[-fu_max:fu_max:du,-fv_max:fv_max:dv]
  u=u.T
  v=v.T
  H1=1j*(np.exp(-1j*(np.dot(np.pi,ri_1**2))*(u**2+v**2)))
  H2=1j*(np.exp(-1j*(np.dot(np.pi,ri_1**2))*(u**2+v**2)+1j*0.5*np.pi))
 # H3=1j*(np.exp(-1j*(np.dot(np.pi,ri_2**2))*(u**2+v**2)))
 # H4=1j*(np.exp(-1j*(np.dot(np.pi,ri_2**2))*(u**2+v**2)+1j*0.5*np.pi))
  H1=np.array(H1,dtype=np.complex128)
  H2=np.array(H2,dtype=np.complex128)
 # H3=np.array(H3,dtype=np.complex128)
 # H4=np.array(H4,dtype=np.complex128)
  #H=torch.tensor(H,dtype=torch.complex128).cuda()

  '''
  img_ob_0 = forward(img[0,0,:,:],H)
  img_ob_1 = forward(img[0,1,:,:],H)
  img_ob_2 = forward(img[0,2,:,:],H)
  img_ob = torch.stack((img_ob_0,img_ob_1,img_ob_2),dim=2)      #(-0.45,0.48)
  #img_ob = img_ob-img_ob.min()
  '''
 # img_ob = (img_ob-img_ob.min())/(img_ob.max()-img_ob.min())              #tensor(256,256,3)
 # print(img.shape,img_ob.shape,img.min(),img.max(),img_ob.min(),img_ob.max())
  '''
  img_forward_0=backward(img_ob[:,:,0],H).cpu().numpy()#(-0.6,0.6)
  img_forward_1=backward(img_ob[:,:,1],H).cpu().numpy()
  img_forward_2=backward(img_ob[:,:,2],H).cpu().numpy()
  img_forward = np.stack((img_forward_0,img_forward_1,img_forward_2),axis=2)
  img_forward = (img_forward-img_forward.min())/(img_forward.max()-img_forward.min())
  '''
  '''
  plt.figure(figsize=(8,8))
  plt.axis('off')
  plt.imshow(img_forward,cmap='gray')
  plt.show()
  '''
  '''
  mask = torch.ones_like(img)

  mask[:, :, :, 128:] = 0.
  show_samples(img * mask)
  '''
  psnr_max_1=0
  for i in range(1):
    print('##################'+str(i)+'#######################')
     
    img_size = config.data.image_size                  #256
    channels = config.data.num_channels               #3
    shape = (batch_size, channels, img_size, img_size)   #(1,3,256,256)

    sampling_fn = sampling_3noise_fujian_mask.get_pc_sampler(sde, shape, predictor, corrector,
                                    inverse_scaler, snr, n_steps=n_steps,
                                    probability_flow=probability_flow,
                                    continuous=config.training.continuous,
                                    eps=sampling_eps, device=config.device)

    #x,psnr_max,ssim_max = sampling_fn(score_model,img,H,img_ob)
    x,psnr_max,ssim_max = sampling_fn(score_model,img,H1,H2,img_ob1,img_ob2)
  #  x = sampling_fn(score_model,img,H1,H2,H3,H4,img_ob1,img_ob2,img_ob3,img_ob4)                          #img:tensor(1,3,256,256)    H:numpy(256,256)  img_ob:tensor(256,256,3)
    '''
    if psnr_max>psnr_max_1:
      psnr_max_1=psnr_max
      ssim_max_1=ssim_max
    '''
    min_1=x.min()
    max_1=x.max()
    x=(x-min_1)/(max_1-min_1)
    '''
    x = torch.as_tensor(x).cuda()
    x_0 = x[:,:,0]+backward(img_ob[:,:,0]-forward(x[:,:,0],H),H)
    x_1 = x[:,:,1]+backward(img_ob[:,:,1]-forward(x[:,:,1],H),H)
    x_2 = x[:,:,2]+backward(img_ob[:,:,2]-forward(x[:,:,2],H),H)
    x = torch.stack((x_0,x_1,x_2),dim=2).cpu().numpy()
    '''
    cv2.imwrite('./house.png',x*255)
    #print('psnr_max_1',psnr_max)
#########################################################################

  psnr_result.append(psnr_max)
  ssim_result.append(ssim_max)
  print('psnr_result',psnr_result)
  print('ssim_result',ssim_result)
psnr_result=sum(psnr_result)/(len(psnr_result))
ssim_result=sum(ssim_result)/(len(ssim_result))
print(psnr_result,ssim_result)

#############################################################################
'''
    x,psnr_max,ssim_max = pc_inpainter(score_model, scaler(img),H,img_ob)
    if psnr_max>psnr_max_1:
      psnr_max_1=psnr_max
      ssim_max_1=ssim_max
      cv2.imwrite('./try/fza_rec_{}.png'.format(j),x*255)
      print('psnr_max_1',psnr_max_1)
  psnr_result.append(psnr_max_1)
  ssim_result.append(ssim_max_1)
  print('psnr_result',psnr_result)
  print('ssim_result',ssim_result)
psnr_result=sum(psnr_result)/(len(psnr_result))
ssim_result=sum(ssim_result)/(len(ssim_result))
print(psnr_result,ssim_result)
'''
      




