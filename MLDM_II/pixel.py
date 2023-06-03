#from pickletools import float8
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

def pixel(img):
    max_1=img.max()
    img=img[:,:]/max_1
    img=np.array(img,dtype=float)
    guibi=np.zeros((256,256),dtype=float)
    inx=-1
    for i in range(0,768,3):
        inx=inx+1
        iny=-1
        for j in range(0,768,3):
            iny=iny+1
            patch=img[i:i+3,j:j+3]
            mean=np.sum(patch)/9
            guibi[inx,iny]=mean
    return guibi



def pixel3(img):
    max_1=img.max()
    min_1=img.min()
    img=(img[:,:,:]-min_1)/(max_1-min_1)*0.3+0.3    #
    #img=img[:,:,:]/max_1
    #img=(img[:,:,:]-min_1)/(max_1-min_1)
    img=np.array(img,dtype=float)
    guibi=np.zeros((256,256,3),dtype=float)
    for k in range(0,3):
        inx=-1
        for i in range(0,768,3):
            inx=inx+1
            iny=-1
            for j in range(0,768,3):
                iny=iny+1
                patch=img[i:i+3,j:j+3,k]
                mean=np.sum(patch)/9
                guibi[inx,iny,k]=mean
    return guibi


def pixel_3(img):
    max_1=img.max()
    min_1=img.min()
    #img=(img[:,:,:]-min_1)/(max_1-min_1)*0.3+0.3    #
    #img=img[:,:,:]/max_1
    img=(img[:,:,:]-min_1)/(max_1-min_1)
    img=np.array(img,dtype=float)
    guibi=np.zeros((256,256,3),dtype=float)
    for k in range(0,3):
        inx=-1
        for i in range(0,768,3):
            inx=inx+1
            iny=-1
            for j in range(0,768,3):
                iny=iny+1
                patch=img[i:i+3,j:j+3,k]
                mean=np.sum(patch)/9
                guibi[inx,iny,k]=mean
    return guibi


