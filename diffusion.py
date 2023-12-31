import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from spectrum import get_spectrum
from torch import Tensor




class MSESpectrumLoss(torch.nn.MSELoss):
    def __init__(self, *args, **kwargs):
        super(MSESpectrumLoss, self).__init__(*args, **kwargs)

    @staticmethod
    def get_log_spectrum(input):
        spectra = get_spectrum(input.flatten(0, 1)).unflatten(0, input.shape[:2])
        spectra = spectra.mean(dim=1)             # average over channels
        return (1 + spectra).log()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input_spectrum = self.get_log_spectrum(input)
        target_spectrum = self.get_log_spectrum(target)
        return super(MSESpectrumLoss, self).forward(input_spectrum, target_spectrum)
    
spectmse = MSESpectrumLoss(reduction='mean')

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss
    

class Skewed_GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T,maxf=100,prob=0.5):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        
        self.maxf = maxf

        self.prob = prob
        
    def sample(self,size,device):

        lst = [1,2]

        prob = self.prob

        
        probs = [prob,1-prob] #Set value of probability

        seed = random.choices(lst,probs,k=1)[0]

        if seed == 2:
            return torch.randint(self.maxf,self.T, size=size, device=device)
        else:
            return torch.randint(0,self.maxf, size=size, device=device)


    def forward(self, x_0):
        """
        Algorithm 1, modified.
        """
        #t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        t = self.sample((x_0.shape[0],),x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss
    

    
class Skewed_GaussianDiffusionTrainer_Spectrum(nn.Module):
    def __init__(self, model, beta_1, beta_T, T,maxf=100,prob=0.5,lamda=1):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        
        self.maxf = maxf

        self.lamda = lamda

        self.prob = prob
        
    def sample(self,size,device):

        lst = [1,2]

        prob = self.prob

        
        probs = [prob,1-prob] #Set value of probability

        seed = random.choices(lst,probs,k=1)[0]

        if seed == 2:
            return (torch.randint(self.maxf,self.T, size=size, device=device),2)
        else:
            return (torch.randint(2,self.maxf, size=size, device=device),1)


    def forward(self, x_0):
        """
        Algorithm 1, modified.
        """
        #t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        t,idx = self.sample((x_0.shape[0],),x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        if idx ==1:
        
            t_p = torch.sub(t,torch.ones_like(t)).to(t.device)
            x_t_prev =  (
            extract(self.sqrt_alphas_bar, t_p, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t_p, x_0.shape) * noise)
            x_t_pred = x_t + (extract(self.sqrt_one_minus_alphas_bar, t_p, x_0.shape) -extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape))*self.model(x_t, t)
            loss_spect = spectmse(x_t_pred,x_t_prev)
            loss = loss + self.lamda*loss_spect
        
        return loss

class Skewed_GaussianDiffusionTrainer_Spectrum_Multistep_Recursive(nn.Module):
    def __init__(self, model, beta_1, beta_T, T,maxf=100,steps=5,prob=0.5,lamda=1):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        
        self.maxf = maxf

        self.steps = steps 

        self.lamda = lamda

        self.prob = prob
        
    def sample(self,size,device,prob):

        lst = [1,2]

        prob = self.prob

        
        probs = [prob,1-prob] #Set value of probability

        seed = random.choices(lst,probs,k=1)[0]

        if seed == 2:
            return (torch.randint(self.maxf,self.T, size=size, device=device),2)
        else:
            return (torch.randint(self.steps,self.maxf, size=size, device=device),1)
        
    def one_step(self,xt,t):
        t_p = torch.sub(t,torch.ones_like(t)).to(t.device)
        xt_prev = xt + (extract(self.sqrt_one_minus_alphas_bar, t_p, xt.shape) -extract(self.sqrt_one_minus_alphas_bar, t, xt.shape))*self.model(xt, t)
        return xt_prev
    
    def extract_gt(self,x0,noise,t,i):
        t_i = torch.sub(t,i*torch.ones_like(t)).to(t.device)
        x_t_prev =  (
            extract(self.sqrt_alphas_bar, t_i, x0.shape) * x0 +
            extract(self.sqrt_one_minus_alphas_bar, t_i, x0.shape) * noise)
        return x_t_prev

    
    def n_step_loss(self,xt,t,x0,noise):
        xgts = [self.extract_gt(x0,noise,t,i) for i in range(self.steps)]
        loss = 0.0
        x_d = xt
        for i in range(self.steps):
            x_d = self.one_step(x_d,t-i)
            loss += spectmse(x_d,xgts[i])
        return loss





    def forward(self, x_0):
        """
        Algorithm 1, modified.
        """
        #t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        t,idx = self.sample((x_0.shape[0],),x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        if idx ==1:
            loss_spect = self.n_step_loss(x_t,t,x_0,noise)
            loss = loss + self.lamda*loss_spect
        
        return loss
    

class Skewed_GaussianDiffusionTrainer_Spectrum_Multistep_Direct(nn.Module):
    def __init__(self, model, beta_1, beta_T, T,maxf=100,steps=5,prob=0.5,lamda=1):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        
        self.maxf = maxf

        self.steps = steps 

        self.lamda = lamda

        self.prob = prob
        
    def sample(self,size,device):

        lst = [1,2]

        prob = self.prob

        
        probs = [prob,1-prob] #Set value of probability

        seed = random.choices(lst,probs,k=1)[0]

        if seed == 2:
            return (torch.randint(self.maxf,self.T, size=size, device=device),2)
        else:
            return (torch.randint(self.steps,self.maxf, size=size, device=device),1)
        
    def i_step(self,xt,t,i):
        t_p = torch.sub(t,i*torch.ones_like(t)).to(t.device)
        xt_prev = xt + (extract(self.sqrt_one_minus_alphas_bar, t_p, xt.shape) -extract(self.sqrt_one_minus_alphas_bar, t, xt.shape))*self.model(xt, t)
        return xt_prev
    
    def extract_gt(self,x0,noise,t,i):
        t_i = torch.sub(t,i*torch.ones_like(t)).to(t.device)
        x_t_prev =  (
            extract(self.sqrt_alphas_bar, t_i, x0.shape) * x0 +
            extract(self.sqrt_one_minus_alphas_bar, t_i, x0.shape) * noise)
        return x_t_prev

    
    def n_step_loss(self,xt,t,x0,noise):
        xgts = [self.extract_gt(x0,noise,t,i) for i in range(self.steps)]
        xps = [self.i_step(xt,t,i) for i in range(self.steps)]
        loss = 0.0
        for i in range(self.steps):
            loss += spectmse(xps[i],xgts[i])
        return loss





    def forward(self, x_0):
        """
        Algorithm 1, modified.
        """
        #t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        t,idx = self.sample((x_0.shape[0],),x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        if idx ==1:
            loss_spect = self.n_step_loss(x_t,t,x_0,noise)
            loss = loss + self.lamda*loss_spect
        
        return loss




class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, img_size=32,
                 mean_type='eps', var_type='fixedlarge'):
        assert mean_type in ['xprev' 'xstart', 'epsilon']
        assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model = model
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )

    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape)

        # Mean parameterization
        if self.mean_type == 'xprev':       # the model predicts x_{t-1}
            x_prev = self.model(x_t, t)
            x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
            model_mean = x_prev
        elif self.mean_type == 'xstart':    # the model predicts x_0
            x_0 = self.model(x_t, t)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        elif self.mean_type == 'epsilon':   # the model predicts epsilon
            eps = self.model(x_t, t)
            x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        else:
            raise NotImplementedError(self.mean_type)
        x_0 = torch.clip(x_0, -1., 1.)

        return model_mean, model_log_var

    def forward(self, x_T):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, log_var = self.p_mean_variance(x_t=x_t, t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.exp(0.5 * log_var) * noise
        x_0 = x_t
        return torch.clip(x_0, -1, 1)
