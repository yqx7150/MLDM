import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.ndimage


di=3
z1=300
x1,y1=0,0
LX1=200
dp=0.014
NX,NY=256,256
S=2*dp*NX
r1=0.23
M=di/z1
ri=(1+M)*r1

def fza1():
    x,y=np.meshgrid(np.linspace(-S/2,S/2-S/(2*NX),(2*NX)),np.linspace(-S/2,S/2-S/(2*NX),(2*NX)))
    r_2=x**2+y**2
    mask = 0.5*(1 + np.cos(np.dot(np.pi,r_2)/ri**2))
    return mask
def fza2():
    x,y=np.meshgrid(np.linspace(-S/2,S/2-S/(2*NX),(2*NX)),np.linspace(-S/2,S/2-S/(2*NX),(2*NX)))
    r_2=x**2+y**2
    mask = 0.5*(1 + np.cos(np.dot(np.pi,r_2)/ri**2+0.5*np.pi))
    return mask
def fza3():
    x,y=np.meshgrid(np.linspace(-S/2,S/2-S/(2*NX),(2*NX)),np.linspace(-S/2,S/2-S/(2*NX),(2*NX)))
    r_2=x**2+y**2
    mask = 0.5*(1 + np.cos(np.dot(np.pi,r_2)/ri**2+np.pi))
    return mask
def fza4():
    x,y=np.meshgrid(np.linspace(-S/2,S/2-S/(2*NX),(2*NX)),np.linspace(-S/2,S/2-S/(2*NX),(2*NX)))
    r_2=x**2+y**2
    mask = 0.5*(1 + np.cos(np.dot(np.pi,r_2)/ri**2+1.5*np.pi))
    return mask




	
	
