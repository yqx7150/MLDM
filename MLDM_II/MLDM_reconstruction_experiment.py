# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""
import functools
from tkinter import W

import torch
import numpy as np
import abc
from operator_fza import forward,backward

from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
import sde_lib
from models import utils as mutils
#from skimage.measure import compare_psnr,compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import cv2
import matplotlib.pyplot as plt
from fza_mask_forward_experiment import fza_mask_forward1, fza_mask_forward2, fza_mask_forward3, fza_mask_forward4
from TV_norm import TVnorm
from tvdenoise import tvdenoise
import scipy.io as io

_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
  """A decorator for registering predictor classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _PREDICTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _PREDICTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def register_corrector(cls=None, *, name=None):
  """A decorator for registering corrector classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _CORRECTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _CORRECTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_predictor(name):
  return _PREDICTORS[name]


def get_corrector(name):
  return _CORRECTORS[name]


def get_sampling_fn(config, sde, shape, inverse_scaler, eps):
  """Create a sampling function.

  Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers representing the expected shape of a single sample.
    inverse_scaler: The inverse data normalizer function.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

  Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
  """

  sampler_name = config.sampling.method
  # Probability flow ODE sampling with black-box ODE solvers
  if sampler_name.lower() == 'ode':
    sampling_fn = get_ode_sampler(sde=sde,
                                  shape=shape,
                                  inverse_scaler=inverse_scaler,
                                  denoise=config.sampling.noise_removal,
                                  eps=eps,
                                  device=config.device)
  # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
  elif sampler_name.lower() == 'pc':
    predictor = get_predictor(config.sampling.predictor.lower())
    corrector = get_corrector(config.sampling.corrector.lower())
    sampling_fn = get_pc_sampler(sde=sde,
                                 shape=shape,
                                 predictor=predictor,
                                 corrector=corrector,
                                 inverse_scaler=inverse_scaler,
                                 snr=config.sampling.snr,
                                 n_steps=config.sampling.n_steps_each,
                                 probability_flow=config.sampling.probability_flow,
                                 continuous=config.training.continuous,
                                 denoise=config.sampling.noise_removal,
                                 eps=eps,
                                 device=config.device)
  else:
    raise ValueError(f"Sampler name {sampler_name} unknown.")

  return sampling_fn


class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__()
    self.sde = sde
    # Compute the reverse SDE/ODE
    self.rsde = sde.reverse(score_fn, probability_flow)
    self.score_fn = score_fn

  @abc.abstractmethod
  def update_fn(self, x, t):
    """One update of the predictor.

    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass


class Corrector(abc.ABC):
  """The abstract class for a corrector algorithm."""

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__()
    self.sde = sde
    self.score_fn = score_fn
    self.snr = snr
    self.n_steps = n_steps

  @abc.abstractmethod
  def update_fn(self, x, t):
    """One update of the corrector.

    Args:
      x: A PyTorch tensor representing the current state
      t: A PyTorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass


@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, x, t):
    dt = -1. / self.rsde.N
    z = torch.randn_like(x)
    drift, diffusion = self.rsde.sde(x, t)
    x_mean = x + drift * dt
    x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
    return x, x_mean


@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, x, t):
    f, G = self.rsde.discretize(x, t)
    z = torch.randn_like(x)
    x_mean = x - f
    x = x_mean + G[:, None, None, None] * z
    return x, x_mean


@register_predictor(name='ancestral_sampling')
class AncestralSamplingPredictor(Predictor):
  """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)
    if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
    assert not probability_flow, "Probability flow not supported by ancestral sampling"

  def vesde_update_fn(self, x, t):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).long()
    sigma = sde.discrete_sigmas[timestep]
    adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), sde.discrete_sigmas.to(t.device)[timestep - 1])
    score = self.score_fn(x, t)
    x_mean = x + score * (sigma ** 2 - adjacent_sigma ** 2)[:, None, None, None]
    std = torch.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
    noise = torch.randn_like(x)
    x = x_mean + std[:, None, None, None] * noise
    return x, x_mean

  def vpsde_update_fn(self, x, t):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).long()
    beta = sde.discrete_betas.to(t.device)[timestep]
    score = self.score_fn(x, t)
    x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1. - beta)[:, None, None, None]
    noise = torch.randn_like(x)
    x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
    return x, x_mean

  def update_fn(self, x, t):
    if isinstance(self.sde, sde_lib.VESDE):
      return self.vesde_update_fn(x, t)
    elif isinstance(self.sde, sde_lib.VPSDE):
      return self.vpsde_update_fn(x, t)


@register_predictor(name='none')
class NonePredictor(Predictor):
  """An empty predictor that does nothing."""

  def __init__(self, sde, score_fn, probability_flow=False):
    pass

  def update_fn(self, x, t):
    return x, x


@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, x1,x2,x3,x_mean, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    for i in range(n_steps):
      grad1 = score_fn(x1, t) # 5 
      grad2 = score_fn(x2, t) # 5 
      grad3 = score_fn(x3, t) # 5 
      
      noise1 = torch.randn_like(x1) # 4 
      noise2 = torch.randn_like(x2) # 4
      noise3 = torch.randn_like(x3) # 4
      
      grad_norm1 = torch.norm(grad1.reshape(grad1.shape[0], -1), dim=-1).mean()
      noise_norm1 = torch.norm(noise1.reshape(noise1.shape[0], -1), dim=-1).mean()
      grad_norm2 = torch.norm(grad2.reshape(grad2.shape[0], -1), dim=-1).mean()
      noise_norm2 = torch.norm(noise2.reshape(noise2.shape[0], -1), dim=-1).mean()      
      grad_norm3 = torch.norm(grad3.reshape(grad3.shape[0], -1), dim=-1).mean()
      noise_norm3 = torch.norm(noise3.reshape(noise3.shape[0], -1), dim=-1).mean()            
      
      grad_norm =(grad_norm1+grad_norm2+grad_norm3)/3.0
      noise_norm = (noise_norm1+noise_norm2+noise_norm3)/3.0
      
      step_size =  (2 * alpha)*((target_snr * noise_norm / grad_norm) ** 2 ) # 6 
   
      x_mean = x_mean + step_size[:, None, None, None] * (grad1+grad2+grad3)/3.0 # 7
      
      x1 = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise1 # 7
      x2 = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise2 # 7
      x3 = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise3 # 7
    return x1,x2,x3, x_mean


@register_corrector(name='ald')
class AnnealedLangevinDynamics(Corrector):
  """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

  We include this corrector only for completeness. It was not directly used in our paper.
  """

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, x, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    std = self.sde.marginal_prob(x, t)[1]

    for i in range(n_steps):
      grad = score_fn(x, t)
      noise = torch.randn_like(x)
      step_size = (target_snr * std) ** 2 * 2 * alpha
      x_mean = x + step_size[:, None, None, None] * grad
      x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

    return x, x_mean


@register_corrector(name='none')
class NoneCorrector(Corrector):
  """An empty corrector that does nothing."""

  def __init__(self, sde, score_fn, snr, n_steps):
    pass

  def update_fn(self, x, t):
    return x, x


def shared_predictor_update_fn(x, t, sde, model, predictor, probability_flow, continuous):
  """A wrapper that configures and returns the update function of predictors."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
  if predictor is None:
    # Corrector-only sampler
    predictor_obj = NonePredictor(sde, score_fn, probability_flow)
  else:
    predictor_obj = predictor(sde, score_fn, probability_flow)
  return predictor_obj.update_fn(x, t)


def shared_corrector_update_fn(x1,x2,x3,x_mean, t, sde, model, corrector, continuous, snr, n_steps):
  """A wrapper tha configures and returns the update function of correctors."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
  if corrector is None:
    # Predictor-only sampler
    corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
  else:
    corrector_obj = corrector(sde, score_fn, snr, n_steps)
  return corrector_obj.update_fn(x1,x2,x3,x_mean, t)


def get_pc_sampler(sde, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda'):
  """Create a Predictor-Corrector (PC) sampler.

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
  # Create predictor & corrector update functions 第一个是函数，其他的是参数
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)

  def pc_sampler(model,data,mask1,mask2,data_ob1,data_ob2):
    """ The PC sampler funciton.

    Args:
      model: A score model.
    Returns:
      Samples, number of function evaluations.
    """
    with torch.no_grad():
      # Initial sample
      x_mean = sde.prior_sampling(shape).to(device)    #(1,3,256,256)
      x1=x_mean
      x2=x_mean
      x3=x_mean
      timesteps = torch.linspace(sde.T, eps, sde.N, device=device)
      psnr_max=0
      ssim_max=0
      psnr_n=[]
      ssim_n=[]
      #for i in range(sde.N):
      for i in range(1000):
        t = timesteps[i]
        vec_t = torch.ones(shape[0], device=t.device) * t
        ########################预测器更新#######################
        x, x_mean = predictor_update_fn(x_mean, vec_t, model=model)#预测器更新    
        #x: A PyTorch tensor of the next state.     
        #x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.

        data_ob_first1=data_ob1.permute(2,0,1).unsqueeze(0)
        data_ob_first2=data_ob2.permute(2,0,1).unsqueeze(0)
        #data_ob_first3=data_ob3.permute(2,0,1).unsqueeze(0)
        #data_ob_first4=data_ob4.permute(2,0,1).unsqueeze(0)

        masked_data_mean1, std = sde.marginal_prob(data_ob_first1, vec_t) #std = self.sigma_min * (self.sigma_max / self.sigma_min) ** vec_t      masked_data_mean = data_ob_first
        masked_data_mean2, std = sde.marginal_prob(data_ob_first2, vec_t)
       # masked_data_mean3, std = sde.marginal_prob(data_ob_first3, vec_t)
       # masked_data_mean4, std = sde.marginal_prob(data_ob_first4, vec_t)

        forward_noise_0,forward_noise_1,forward_noise_2=fza_mask_forward1(torch.randn_like(x_mean))      #(256,256)
        noise_result1=torch.stack((forward_noise_0,forward_noise_1,forward_noise_2),dim=0).unsqueeze(0)   #(1,3,256,256)
        masked_data1 = masked_data_mean1 +  noise_result1* std[:, None, None, None]

        forward_noise_0,forward_noise_1,forward_noise_2=fza_mask_forward2(torch.randn_like(x_mean))      #(256,256)
        noise_result2=torch.stack((forward_noise_0,forward_noise_1,forward_noise_2),dim=0).unsqueeze(0)   #(1,3,256,256)
        masked_data2 = masked_data_mean2 +  noise_result2* std[:, None, None, None]
        '''
        forward_noise_0,forward_noise_1,forward_noise_2=fza_mask_forward3(torch.randn_like(x_mean))      #(256,256)
        noise_result3=torch.stack((forward_noise_0,forward_noise_1,forward_noise_2),dim=0).unsqueeze(0)   #(1,3,256,256)
        masked_data3 = masked_data_mean3 +  noise_result3* std[:, None, None, None]

        forward_noise_0,forward_noise_1,forward_noise_2=fza_mask_forward4(torch.randn_like(x_mean))      #(256,256)
        noise_result4=torch.stack((forward_noise_0,forward_noise_1,forward_noise_2),dim=0).unsqueeze(0)   #(1,3,256,256)
        masked_data4 = masked_data_mean4 +  noise_result4* std[:, None, None, None]
        '''

        a1=0.5    #0.05
      #  a2=0
        b1=0.5
      #  b2=0

        c=1
        d=1

        w1=0.5
     #   w2=0
        k1=0.5
     #   k2=0
        #############################x_mean#################################
        x_red=c*x_mean[0,0,:,:]+a1*backward(masked_data1[0,0,:,:]-forward(x_mean[0,0,:,:],mask1),mask1)+b1*backward(masked_data2[0,0,:,:]-forward(x_mean[0,0,:,:],mask2),mask2)
         # +a2*backward(masked_data3[0,0,:,:]-forward(x_mean[0,0,:,:],mask3),mask3)+b2*backward(masked_data4[0,0,:,:]-forward(x_mean[0,0,:,:],mask4),mask4)
        x_green=c*x_mean[0,1,:,:]+a1*backward(masked_data1[0,1,:,:]-forward(x_mean[0,1,:,:],mask1),mask1)+b1*backward(masked_data2[0,1,:,:]-forward(x_mean[0,1,:,:],mask2),mask2)
         # +a2*backward(masked_data3[0,1,:,:]-forward(x_mean[0,1,:,:],mask3),mask3)+b2*backward(masked_data4[0,1,:,:]-forward(x_mean[0,1,:,:],mask4),mask4)
        x_blue=c*x_mean[0,2,:,:]+a1*backward(masked_data1[0,2,:,:]-forward(x_mean[0,2,:,:],mask1),mask1)+b1*backward(masked_data2[0,2,:,:]-forward(x_mean[0,2,:,:],mask2),mask2)
         # +a2*backward(masked_data3[0,2,:,:]-forward(x_mean[0,2,:,:],mask3),mask3)+b2*backward(masked_data4[0,2,:,:]-forward(x_mean[0,2,:,:],mask4),mask4)
        x_mean=torch.stack((x_red,x_green,x_blue),dim=0).unsqueeze(0)

        ###############################x####################################
        x_red=d*x_mean[0,0,:,:]+w1*backward(masked_data_mean1[0,0,:,:]-forward(x_mean[0,0,:,:],mask1),mask1)+k1*backward(masked_data_mean2[0,0,:,:]-forward(x_mean[0,0,:,:],mask2),mask2)
         # +w2*backward(masked_data_mean3[0,0,:,:]-forward(x_mean[0,0,:,:],mask3),mask3)+k2*backward(masked_data_mean4[0,0,:,:]-forward(x_mean[0,0,:,:],mask4),mask4)
        x_green=d*x_mean[0,1,:,:]+w1*backward(masked_data_mean1[0,1,:,:]-forward(x_mean[0,1,:,:],mask1),mask1)+k1*backward(masked_data_mean2[0,1,:,:]-forward(x_mean[0,1,:,:],mask2),mask2)
         # +w2*backward(masked_data_mean3[0,1,:,:]-forward(x_mean[0,1,:,:],mask3),mask3)+k2*backward(masked_data_mean4[0,1,:,:]-forward(x_mean[0,1,:,:],mask4),mask4)
        x_blue=d*x_mean[0,2,:,:]+w1*backward(masked_data_mean1[0,2,:,:]-forward(x_mean[0,2,:,:],mask1),mask1)+k1*backward(masked_data_mean2[0,2,:,:]-forward(x_mean[0,2,:,:],mask2),mask2)
         # +w2*backward(masked_data_mean3[0,2,:,:]-forward(x_mean[0,2,:,:],mask3),mask3)+k2*backward(masked_data_mean4[0,2,:,:]-forward(x_mean[0,2,:,:],mask4),mask4)
        x=torch.stack((x_red,x_green,x_blue),dim=0).unsqueeze(0)
        x1,x2,x3=x,x,x
    

        ########################校正器更新#########################
        x1,x2,x3, x_mean = corrector_update_fn(x1,x2,x3,x_mean, vec_t, model=model)#校正器更新
        
        #############################x_mean#################################
        x_red=c*x_mean[0,0,:,:]+a1*backward(masked_data1[0,0,:,:]-forward(x_mean[0,0,:,:],mask1),mask1)+b1*backward(masked_data2[0,0,:,:]-forward(x_mean[0,0,:,:],mask2),mask2)
         # +a2*backward(masked_data3[0,0,:,:]-forward(x_mean[0,0,:,:],mask3),mask3)+b2*backward(masked_data4[0,0,:,:]-forward(x_mean[0,0,:,:],mask4),mask4)
        x_green=c*x_mean[0,1,:,:]+a1*backward(masked_data1[0,1,:,:]-forward(x_mean[0,1,:,:],mask1),mask1)+b1*backward(masked_data2[0,1,:,:]-forward(x_mean[0,1,:,:],mask2),mask2)
         # +a2*backward(masked_data3[0,1,:,:]-forward(x_mean[0,1,:,:],mask3),mask3)+b2*backward(masked_data4[0,1,:,:]-forward(x_mean[0,1,:,:],mask4),mask4)
        x_blue=c*x_mean[0,2,:,:]+a1*backward(masked_data1[0,2,:,:]-forward(x_mean[0,2,:,:],mask1),mask1)+b1*backward(masked_data2[0,2,:,:]-forward(x_mean[0,2,:,:],mask2),mask2)
         # +a2*backward(masked_data3[0,2,:,:]-forward(x_mean[0,2,:,:],mask3),mask3)+b2*backward(masked_data4[0,2,:,:]-forward(x_mean[0,2,:,:],mask4),mask4)
        x_mean=torch.stack((x_red,x_green,x_blue),dim=0).unsqueeze(0)

        ###############################x####################################
        x_red=d*x_mean[0,0,:,:]+w1*backward(masked_data_mean1[0,0,:,:]-forward(x_mean[0,0,:,:],mask1),mask1)+k1*backward(masked_data_mean2[0,0,:,:]-forward(x_mean[0,0,:,:],mask2),mask2)
         # +w2*backward(masked_data_mean3[0,0,:,:]-forward(x_mean[0,0,:,:],mask3),mask3)+k2*backward(masked_data_mean4[0,0,:,:]-forward(x_mean[0,0,:,:],mask4),mask4)
        x_green=d*x_mean[0,1,:,:]+w1*backward(masked_data_mean1[0,1,:,:]-forward(x_mean[0,1,:,:],mask1),mask1)+k1*backward(masked_data_mean2[0,1,:,:]-forward(x_mean[0,1,:,:],mask2),mask2)
         # +w2*backward(masked_data_mean3[0,1,:,:]-forward(x_mean[0,1,:,:],mask3),mask3)+k2*backward(masked_data_mean4[0,1,:,:]-forward(x_mean[0,1,:,:],mask4),mask4)
        x_blue=d*x_mean[0,2,:,:]+w1*backward(masked_data_mean1[0,2,:,:]-forward(x_mean[0,2,:,:],mask1),mask1)+k1*backward(masked_data_mean2[0,2,:,:]-forward(x_mean[0,2,:,:],mask2),mask2)
         # +w2*backward(masked_data_mean3[0,2,:,:]-forward(x_mean[0,2,:,:],mask3),mask3)+k2*backward(masked_data_mean4[0,2,:,:]-forward(x_mean[0,2,:,:],mask4),mask4)
        x=torch.stack((x_red,x_green,x_blue),dim=0).unsqueeze(0)
      
        '''
        x_0=tvdenoise(x[0,0,:,:],10,2)#TV去噪
        x_1=tvdenoise(x[0,1,:,:],10,2)
        x_2=tvdenoise(x[0,2,:,:],10,2)
        x=torch.stack((x_0,x_1,x_2),dim=0).unsqueeze(0)
        '''

        x_mean_save=x.cpu().numpy()
        x_mean_save_cv=x_mean_save[0,:,:,:].transpose(1,2,0)
        data_ori=data[0,:,:,:].cpu().numpy().transpose(1,2,0)
        print(x_mean_save_cv.min(),x_mean_save_cv.max())
       # x_mean_save_cv[57:200,57:200,:] = x_mean_save_cv[57:200,57:200,:]-x_mean_save_cv[57:200,57:200,:].min()
       # x_mean_save_cv[57:200,57:200,:] = (x_mean_save_cv[57:200,57:200,:]-x_mean_save_cv[57:200,57:200,:].min())/(x_mean_save_cv[57:200,57:200,:].max()-x_mean_save_cv[57:200,57:200,:].min())  ########
        x_mean_save_cv = (x_mean_save_cv-x_mean_save_cv.min())/(x_mean_save_cv.max()-x_mean_save_cv.min())      ##############################
        data_ori = (data_ori-data_ori.min())/(data_ori.max()-data_ori.min())
        x_mean_save_cv=np.clip(x_mean_save_cv,0,1)
        data_ori=np.clip(data_ori,0,1)

        psnr=compare_psnr(np.abs(x_mean_save_cv)*255,np.abs(data_ori)*255,data_range=255)
        ssim=compare_ssim(np.abs(x_mean_save_cv),np.abs(data_ori),multichannel=True,data_range=1)
       
       # x_mean_save_img=(x_mean_save_cv[:,:,0]+x_mean_save_cv[:,:,1]+x_mean_save_cv[:,:,2])/3
       # x_mean_save_img_1=(x_mean_save_cv[:,:,0]+x_mean_save_cv[:,:,1]+x_mean_save_cv[:,:,2])/3
        x_mean_save_img=np.stack((x_mean_save_cv[:,:,2],x_mean_save_cv[:,:,1],x_mean_save_cv[:,:,0]),axis=2)        ######################
        cv2.imwrite('Real-time reconstruction.png',x_mean_save_img*255)                                    ###########################
        print('itertions:',i,'psnr:',psnr,'ssim:',ssim)
        psnr_n.append(psnr)
        ssim_n.append(ssim)
        if psnr>psnr_max:
          psnr_max=psnr
          ssim_max=ssim
          print('psnr_max',psnr_max)
         # x_mean_save_img=np.stack((x_mean_save_cv[:,:,2],x_mean_save_cv[:,:,1],x_mean_save_cv[:,:,0]),axis=2)
          #x_mean_save_img=(x_mean_save_cv[:,:,0]+x_mean_save_cv[:,:,1]+x_mean_save_cv[:,:,2])/3
         # cv2.imwrite('fza_inpainting_3noise_0.2_FOV_z=12.5.png',x_mean_save_img*255)                        ####################################
    #  print('psnr_n',psnr_n)
    #  print('ssim_n',ssim_n)
      print('psnr_max:',psnr_max,'ssim_max:',ssim_max)
      return x_mean_save_img,psnr_max,ssim_max

  return pc_sampler


