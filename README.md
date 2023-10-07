# MLDM

**Paper**: Multi-phase FZA Lensless Imaging via Diffusion Model (MLDM)

**Authors**: Wenbo Wan, Huihui Ma, Zijie Mei, Huilin Zhou, Yuhao Wang, Senior Member, IEEE, Qiegen Liu, Senior Member, IEEE

Optics EXPRESS [https://opg.optica.org/oe/fulltext.cfm?uri=oe-31-12-20595&id=531211]

Date : June-2-2023  
Version : 1.0  
The code and the algorithm are for non-comercial use only.  
Copyright 2023, Department of Electronic Information Engineering, Nanchang University. 

Lensless imaging shifts the burden of imaging from bulky and expensive hardware to computing, which enables new architectures for portable cameras. However, the twin image effect caused by the missing phase information in the light wave is a key factor limiting the quality of lensless imaging. Conventional single-phase encoding methods and independent reconstruction of separate channels pose challenges in removing twin images and preserving the color fidelity of the reconstructed image. In order to achieve high-quality lensless imaging, the multiphase lensless imaging via diffusion model (MLDM) is proposed. A multi-phase FZA encoder integrated on a single mask plate is used to expand the data channel of a single-shot image. The information association between the color image pixel channel and the encoded phase channel is established by extracting prior information of the data distribution based on multi-channel encoding. Finally, the reconstruction quality is improved through the use of the iterative reconstruction method. The results show that the proposed MLDM method effectively removes the influence of twin images and produces high-quality reconstructed images compared with traditional methods, and the results reconstructed using MLDM have higher structural similarity and peak signal-to-noise ratio.


## The main processes of lensless imaging include optical encoding and computer decoding processes.
<div align="center"><img src="https://github.com/yqx7150/MLDM/blob/main/Figs/fig1.png"> </div>

## Reconstruction flow chart of MLDM-I algorithm.
<div align="center"><img src="https://github.com/yqx7150/MLDM/tree/main/Figs/fig3.png"> </div>

## Reconstruction flow chart of MLDM-II algorithm.
<div align="center"><img src="https://github.com/yqx7150/MLDM/tree/main/Figs/fig5.png"> </div>

## The multi-phase lensless imaging system.
<div align="center"><img src="https://github.com/yqx7150/MLDM/tree/main/Figs/fig8.png"> </div>

## Visual comparison of reconstructed images on optical experiment. (a) Ground Truth (b) BP (c) CS (d) LSGM (e) ADMM (f) MLDM-I (g) MLDM-II.
<div align="center"><img src="https://github.com/yqx7150/MLDM/tree/main/Figs/fig9.png"> </div>


## Requirements and Dependencies
    python==3.7.11
    Pytorch==1.7.0
    tensorflow==2.4.0
    torchvision==0.8.0
    tensorboard==2.7.0
    scipy==1.7.3
    numpy==1.19.5
    ninja==1.10.2
    matplotlib==3.5.1
    jax==0.2.26

## Checkpoints
MLDM_I : We provide pretrained checkpoints. You can download pretrained models from  [Baidu cloud] (https://pan.baidu.com/s/1CX7xCh1uJl-h5ZojO5SkNQ?pwd=hdtp) Extract the code (hdtp)

MLDM_II : We provide pretrained checkpoints. You can download pretrained models from  [Baidu cloud] (https://pan.baidu.com/s/1-0WZnOfjQ5eiTjk1aO93PQ?pwd=taf4) Extract the code (taf4)

## Dataset

The dataset used to train the model in this experiment is  LSUN-bedroom and  LSUN-church.

place the dataset in the train file under the church folder.

## Train:

python main.py --config=configs/ve/church_ncsnpp_continuous.py  --workdir=exp_train_church_max1_N1000 --mode=train --eval_folder=result


## Test:

Simulate : python MLDM_reconstruction_simulate.py

Experiment : python MLDM_reconstruction_experiment.py


## Acknowledgement
The implementation is based on this repository: https://github.com/yang-song/score_sde_pytorch.


## Other Related Projects
  * Diffusion Models for Medical Imaging
[<font size=5>**[Paper]**</font>](https://github.com/yqx7150/Diffusion-Models-for-Medical-Imaging)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/Diffusion-Models-for-Medical-Imaging)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HKGM/tree/main/PPT)  
  * Homotopic Gradients of Generative Density Priors for MR Image Reconstruction  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/9435335)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/HGGDP)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)
  * Multi-Channel and Multi-Model-Based Autoencoding Prior for Grayscale Image Restoration  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8782831)  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/MEDAEP)   [<font size=5>**[Slide]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)  [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

  * Highly Undersampled Magnetic Resonance Imaging Reconstruction using Autoencoding Priors  
[<font size=5>**[Paper]**</font>](https://cardiacmr.hms.harvard.edu/files/cardiacmr/files/liu2019.pdf)  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EDAEPRec)   [<font size=5>**[Slide]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide) [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

  * High-dimensional Embedding Network Derived Prior for Compressive Sensing MRI Reconstruction  
 [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S1361841520300815?via%3Dihub)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EDMSPRec)
 
  * Denoising Auto-encoding Priors in Undecimated Wavelet Domain for MR Image Reconstruction  
[<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/pii/S0925231221000990) [<font size=5>**[Paper]**</font>](https://arxiv.org/ftp/arxiv/papers/1909/1909.01108.pdf)  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/WDAEPRec)
  
  * Learning Multi-Denoising Autoencoding Priors for Image Super-Resolution  
[<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/pii/S1047320318302700)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/MDAEP-SR)

  * REDAEP: Robust and Enhanced Denoising Autoencoding Prior for Sparse-View CT Reconstruction  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/9076295)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/REDAEP)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)  [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

  * Iterative Reconstruction for Low-Dose CT using Deep Gradient Priors of Generative Model  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/9703672)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EASEL)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)

  * Universal Generative Modeling for Calibration-free Parallel MR Imaging  
[<font size=5>**[Paper]**</font>](https://biomedicalimaging.org/2022/)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/UGM-PI)   [<font size=5>**[Poster]**</font>](https://github.com/yqx7150/UGM-PI/blob/main/paper%20%23160-Poster.pdf)

* Progressive Colorization via Interative Generative Models  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/9258392)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/iGM)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)  [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

* Joint Intensity-Gradient Guided Generative Modeling for Colorization
[<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2012.14130)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/JGM)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)  [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)
