from fza_mask_experiment import fza1, fza2, fza3, fza4
import torch
import numpy as np
import cv2

di=3.4
z1=500
LX1=200
dp=0.0038*3
NX,NY=256,256
S=2*dp*NX
r1=0.4

def fza_mask_forward1(img):
  masked_data_0=img[0,0,:,:].detach().cpu().numpy()
  masked_data_1=img[0,1,:,:].detach().cpu().numpy()
  masked_data_2=img[0,2,:,:].detach().cpu().numpy()

  M=di/z1
  ri=(1+M)*r1
  I_0=cv2.filter2D(masked_data_0.astype('float32'), -1, fza1(), borderType=cv2.BORDER_CONSTANT)*2*dp*dp/ri**2
  I_1=cv2.filter2D(masked_data_1.astype('float32'), -1, fza1(), borderType=cv2.BORDER_CONSTANT)*2*dp*dp/ri**2
  I_2=cv2.filter2D(masked_data_2.astype('float32'), -1, fza1(), borderType=cv2.BORDER_CONSTANT)*2*dp*dp/ri**2
  forward_fza_0=I_0-np.mean(I_0[:])
  forward_fza_1=I_1-np.mean(I_1[:])
  forward_fza_2=I_2-np.mean(I_2[:])
  forward_fza_0=torch.as_tensor(forward_fza_0).cuda()
  forward_fza_1=torch.as_tensor(forward_fza_1).cuda()
  forward_fza_2=torch.as_tensor(forward_fza_2).cuda()
  
  return forward_fza_0,forward_fza_1,forward_fza_2

def fza_mask_forward2(img):
  masked_data_0=img[0,0,:,:].detach().cpu().numpy()
  masked_data_1=img[0,1,:,:].detach().cpu().numpy()
  masked_data_2=img[0,2,:,:].detach().cpu().numpy()

  M=di/z1
  ri=(1+M)*r1
  I_0=cv2.filter2D(masked_data_0.astype('float32'), -1, fza2(), borderType=cv2.BORDER_CONSTANT)*2*dp*dp/ri**2
  I_1=cv2.filter2D(masked_data_1.astype('float32'), -1, fza2(), borderType=cv2.BORDER_CONSTANT)*2*dp*dp/ri**2
  I_2=cv2.filter2D(masked_data_2.astype('float32'), -1, fza2(), borderType=cv2.BORDER_CONSTANT)*2*dp*dp/ri**2
  forward_fza_0=I_0-np.mean(I_0[:])
  forward_fza_1=I_1-np.mean(I_1[:])
  forward_fza_2=I_2-np.mean(I_2[:])
  forward_fza_0=torch.as_tensor(forward_fza_0).cuda()
  forward_fza_1=torch.as_tensor(forward_fza_1).cuda()
  forward_fza_2=torch.as_tensor(forward_fza_2).cuda()
  
  return forward_fza_0,forward_fza_1,forward_fza_2

def fza_mask_forward3(img):
  masked_data_0=img[0,0,:,:].detach().cpu().numpy()
  masked_data_1=img[0,1,:,:].detach().cpu().numpy()
  masked_data_2=img[0,2,:,:].detach().cpu().numpy()

  M=di/z1
  ri=(1+M)*r1
  I_0=cv2.filter2D(masked_data_0.astype('float32'), -1, fza3(), borderType=cv2.BORDER_CONSTANT)*2*dp*dp/ri**2
  I_1=cv2.filter2D(masked_data_1.astype('float32'), -1, fza3(), borderType=cv2.BORDER_CONSTANT)*2*dp*dp/ri**2
  I_2=cv2.filter2D(masked_data_2.astype('float32'), -1, fza3(), borderType=cv2.BORDER_CONSTANT)*2*dp*dp/ri**2
  forward_fza_0=I_0-np.mean(I_0[:])
  forward_fza_1=I_1-np.mean(I_1[:])
  forward_fza_2=I_2-np.mean(I_2[:])
  forward_fza_0=torch.as_tensor(forward_fza_0).cuda()
  forward_fza_1=torch.as_tensor(forward_fza_1).cuda()
  forward_fza_2=torch.as_tensor(forward_fza_2).cuda()
  
  return forward_fza_0,forward_fza_1,forward_fza_2

def fza_mask_forward4(img):
  masked_data_0=img[0,0,:,:].detach().cpu().numpy()
  masked_data_1=img[0,1,:,:].detach().cpu().numpy()
  masked_data_2=img[0,2,:,:].detach().cpu().numpy()

  M=di/z1
  ri=(1+M)*r1
  I_0=cv2.filter2D(masked_data_0.astype('float32'), -1, fza4(), borderType=cv2.BORDER_CONSTANT)*2*dp*dp/ri**2
  I_1=cv2.filter2D(masked_data_1.astype('float32'), -1, fza4(), borderType=cv2.BORDER_CONSTANT)*2*dp*dp/ri**2
  I_2=cv2.filter2D(masked_data_2.astype('float32'), -1, fza4(), borderType=cv2.BORDER_CONSTANT)*2*dp*dp/ri**2
  forward_fza_0=I_0-np.mean(I_0[:])
  forward_fza_1=I_1-np.mean(I_1[:])
  forward_fza_2=I_2-np.mean(I_2[:])
  forward_fza_0=torch.as_tensor(forward_fza_0).cuda()
  forward_fza_1=torch.as_tensor(forward_fza_1).cuda()
  forward_fza_2=torch.as_tensor(forward_fza_2).cuda()
  
  return forward_fza_0,forward_fza_1,forward_fza_2
