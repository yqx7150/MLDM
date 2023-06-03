from torch.utils.data import DataLoader, Subset,Dataset
import torchvision.transforms as transforms
import torchvision.transforms as T
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
class LoadDataset(Dataset):
    def __init__(self,file_list,transform=None):
        self.files = glob.glob(file_list+ '/*.png')  #church
        '''
        self.folders = glob.glob(file_list+ '/*')
        self.files=[]
        for folder in self.folders:
            for f in glob.glob(folder+'/*.png'):
                self.files.append(f)
        '''
        
        
        self.transform = transform
        self.to_tensor = T.ToTensor()
    def __len__(self):
        return len(self.files)
    def __getitem__(self,index):
        img = cv2.imread(self.files[index], 3)
        img = cv2.resize(img, (256, 256))
        im = img[:,:,:3]/255
        im=np.stack((im[:,:,2],im[:,:,1],im[:,:,0],im[:,:,2],im[:,:,1],im[:,:,0]),axis=2)
        '''
        im_r_fft=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(im[:,:,0])))
        im_g_fft=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(im[:,:,1])))
        im_b_fft=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(im[:,:,2])))
        img_all=np.stack((np.real(im_r_fft),np.imag(im_r_fft),np.real(im_g_fft),np.imag(im_g_fft),np.real(im_b_fft),np.imag(im_b_fft)),axis=2)
        '''
        return im.transpose(2,0,1)  #img_all.transpose(2,0,1)
