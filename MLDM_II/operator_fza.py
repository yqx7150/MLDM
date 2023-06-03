import numpy as np
import torch.fft as fft
import torch
def forward(obj,H):
        obj=obj.cpu().numpy()
        FO=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(obj)))
        #FO=np.array(FO,dtype=np.complex64)
        I=np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(FO*H)))
        I=np.real(I)
        I=np.array(I,dtype=np.float32)
        I=torch.as_tensor(I).cuda()
        return I
def backward(I,H):
	I=I.cpu().numpy()
	FI=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(I)))#0.01
	#FI=np.array(FI,dtype=np.complex64)
	OR=np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(FI/H)))
	OR=np.real(OR)
	OR=np.array(OR,dtype=np.float32)
	OR=torch.as_tensor(OR).cuda()
	return OR
def forward_torch(obj,H):
        FO=fft.fftn(obj)
        I=fft.ifftn(FO*H)
        I=torch.real(I)
        return I
def backward_torch(I,H):
	FI=fft.fftn(I)#0.01
	OR=fft.ifftn(FI/H)
	OR=torch.real(OR)
	return OR
