from functools import partial
from typing import *

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.functional import softmax
import torch.nn.functional as F

from torch.distributions import Normal
from models.unet import *
from models.nn import *

torch.set_printoptions(precision=3)


# Sample from the Gumbel-Softmax distribution and optionally discretize.
class GumbelSoftmax(nn.Module):

    def __init__(self, c_dim, temperature=1.0, hard=False):
        super(GumbelSoftmax, self).__init__()
        # self.logits = nn.Linear(f_dim, c_dim)
        # self.f_dim = f_dim
        self.c_dim = c_dim
        self.temperature = temperature
        self.hard = hard
        # self.device = device
     
    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / self.temperature, dim=-1)

    def sample(self, logits):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        #categorical_dim = 10
        y = self.gumbel_softmax_sample(logits)

        if not self.hard:
            return y

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard 

    def forward(self, logits):
        # logits = self.logits(x).view(-1, self.c_dim)
        prob = F.softmax(logits, dim=-1)
        # y = self.sample(logits)
        # return logits, prob, y
        return prob


class GaussianMixtureComponent(nn.Module):
    def __init__(self, K, z_features, x_features, freqs=2, hidden_dim=784, in_ch=3, mod_ch=8, **kwargs):
        super().__init__()

        self.K = K
        self.hidden_dim = hidden_dim

        self.embx = nn.Sequential(
            # nn.LayerNorm(x_features),
            # nn.Linear(x_features, hidden_dim),
            # nn.LayerNorm(hidden_dim)
            # nn.Tanh()
        )
        
        self.embpi = nn.Sequential(
            # nn.Linear(K, hidden_dim),
            # nn.Tanh()
            # nn.LayerNorm(hidden_dim)
        )
        self.hyper =  nn.Sequential(
            # MLP(2*x_features+2*freqs, z_features*2, **kwargs),
            MLP(x_features + hidden_dim+2*freqs, z_features*2, **kwargs),
            # MLP(2*hidden_dim+2*freqs, z_features*2, **kwargs),
            # MLP(2*hidden_dim+2*freqs, z_features*2, **kwargs),
            # MLP(x_features + K + 2*freqs, z_features*2, **kwargs),
            # nn.Tanh()
            )
        self.z_features = z_features
        self.register_buffer('freqs', torch.arange(1, freqs + 1) * torch.pi)
        
    def forward(self, pi:Tensor, x:Tensor, t:Tensor):
        x = x.flatten(start_dim=1) # if x.ndim >= 4 else x
        t = self.freqs * t[..., None]
        temb = torch.cat((t.cos(), t.sin()), dim=-1)
        temb = temb.expand(*pi.shape[:-1], -1)
        xemb = self.embx(x)
        pi_ordered, _ = torch.sort(pi, dim=-1, descending=True)  # [B, K]
        piemb = self.embpi(pi_ordered)
        phi = self.hyper(torch.cat((temb, piemb, xemb), dim=-1))

        mu, sigma = phi.chunk(2, dim=-1)
        return Normal(mu, F.softplus(sigma) + 1e-8)

    def rsample(self, pi:Tensor, x:Tensor,t:Tensor=None):
        if t is None:
            t = torch.tensor(0., device=x.device)
        dist = self(pi, x, t)
        return dist.rsample()

    def sample(self, pi:Tensor, x:Tensor, t:Tensor=None):
        if t is None:
            t = torch.tensor(0., device=x.device)
        dist = self(pi, x, t)
        return dist.sample()

    def log_prob(self, pi:Tensor, x:Tensor, z:Tensor, t:Tensor=None):
        if t is None:
            t = torch.tensor(0., device=x.device)
        dist = self(pi, x, t)
        return dist.log_prob(z).sum(-1)

# class GaussianMixtureComponent(nn.Module):
#     def __init__(self, K, z_features, x_features, freqs=2, hidden_dim=784, in_ch=3, mod_ch=8, **kwargs):
#         super().__init__()
#         self.K = K

#         # x embedding
#         self.embx = nn.Sequential(
#             nn.LayerNorm(x_features),
#             # nn.Linear(x_features, hidden_dim),
#             # nn.SiLU(),
#         )

#         # pi → FiLM params (scale + shift for each hidden dim)
#         self.film = nn.Sequential(
#             nn.Linear(K, hidden_dim * 2),   # outputs (γ, β)
#         )

#         # final projection to (mu, logvar)
#         self.hyper = nn.Sequential(
#             MLP(x_features + 2*freqs, z_features*2, **kwargs),
#         )

#         self.z_features = z_features
#         self.register_buffer('freqs', torch.arange(1, freqs + 1) * torch.pi)
#         nn.init.zeros_(self.film[-1].weight)
#         nn.init.constant_(self.film[-1].bias, 0)
#         # manually set gamma bias to 1 (identity scale)
#         self.film[-1].bias.data[:hidden_dim] = 1.0

#     def forward(self, pi: Tensor, x: Tensor, t: Tensor):
#         x = x.flatten(start_dim=1)
#         t = self.freqs * t[..., None]
#         temb = torch.cat((t.cos(), t.sin()), dim=-1)
#         temb = temb.expand(*pi.shape[:-1], -1)

#         # sort for label stability
#         pi_ordered, _ = torch.sort(pi, dim=-1, descending=True)

#         # embed x
#         xemb = self.embx(x)                          # [B, hidden_dim]

#         # FiLM: pi modulates xemb
#         film_params = self.film(pi_ordered)           # [B, hidden_dim*2]
#         gamma, beta = film_params.chunk(2, dim=-1)    # [B, hidden_dim] each
#         xemb = gamma * xemb + beta                    # pi gates x features

#         phi = self.hyper(torch.cat((temb, xemb), dim=-1))
#         mu, sigma = phi.chunk(2, dim=-1)
#         return Normal(mu, F.softplus(sigma) + 1e-8)
    
#     def rsample(self, pi:Tensor, x:Tensor,t:Tensor=None):
#         if t is None:
#             t = torch.tensor(0., device=x.device)
#         dist = self(pi, x, t)
#         return dist.rsample()

#     def sample(self, pi:Tensor, x:Tensor, t:Tensor=None):
#         if t is None:
#             t = torch.tensor(0., device=x.device)
#         dist = self(pi, x, t)
#         return dist.sample()

#     def log_prob(self, pi:Tensor, x:Tensor, z:Tensor, t:Tensor=None):
#         if t is None:
#             t = torch.tensor(0., device=x.device)
#         dist = self(pi, x, t)
#         return dist.log_prob(z).sum(-1)

class GaussianMixturePrior(nn.Module):
    def __init__(self, K, z_features, **kwargs):
        super().__init__()

        self.K = K
        self.hyper =  nn.Sequential(
            MLP(K, z_features, **kwargs)
            )
        self.z_features = z_features
        
    def forward(self, pi:Tensor):
        phi = self.hyper(pi)
        return Normal(phi, torch.ones(self.z_features).to(pi.device) * .1)

    def sample(self, pi, size):
        return self(pi).sample(size)[0]

    def rsample(self, pi, size):
        return self(pi).rsample(size)[0]

    def log_prob(self, pi, z):
        return self(pi).log_prob(z).sum(-1)


class CatNF_fixed(nn.Module):
    def __init__(self, x_features: int, k: int, temp=5., hard=False, in_ch=3, freqs=2, mod_ch=8, **kwargs):
        super(CatNF_fixed, self).__init__()
        self.k = k
        self.temp = temp
        self.hard = hard
        self.register_buffer('freqs_t', torch.arange(1, freqs + 1) * torch.pi)

        self.embx = nn.Sequential(
            # nn.LayerNorm(x_features),
            )
 
        self.fc = nn.Sequential(
            MLP(2*freqs+x_features, k, **kwargs),
            # nn.LayerNorm(k)
            # nn.Tanh()
        )

        self.register_buffer('freqs', torch.arange(1, freqs + 1) * torch.pi)
        
    def _sample_gumbel(self, shape, device, eps=1e-20):
        U = torch.rand(shape, device=device)
        return -torch.log(-torch.log(U + eps) + eps)

    def _gumbel_softmax_sample(self, logits):
        y = logits + self._sample_gumbel(logits.size(), logits.device)
        return F.softmax(y / self.temp, dim=-1)

    def _sample(self, logits):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        #categorical_dim = 10
        y = self._gumbel_softmax_sample(logits)

        if not self.hard:
            return y

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard 

    def _forward(self, logits):
        prob = F.softmax(logits, dim=-1)
        return prob

    def forward(self, x: Tensor, t: Tensor):
        t = self.freqs * t[..., None]
        temb = torch.cat((t.cos(), t.sin()), dim=-1)
        
        xemb = self.embx(x.flatten(start_dim=1))
        temb = temb.expand(*xemb.shape[:-1], -1)
        h = self.fc(torch.cat((temb, xemb), dim=-1))

        return self._forward(h)

    def rsample(self, x: Tensor, t:Tensor, logits=None) -> Tensor:
        t = self.freqs * t[..., None]
        temb = torch.cat((t.cos(), t.sin()), dim=-1)

        if x is None and logits is not None:
            zt = self._sample(logits)
        else:
            xemb = self.embx(x.flatten(start_dim=1))
            temb = temb.expand(*xemb.shape[:-1], -1)
            logits = self.fc(torch.cat((temb, xemb), dim=-1))
            zt = self._sample(logits)
        return logits, zt



class FlowMatchingLossMixture(nn.Module):
    def __init__(self, vt: nn.Module, Rt: nn.Module, rt, priorpi, priorz,
                 priory=None, k: int=None,
                 sig_min=1e-4, beta=0.1, alpha=0.1, tau=1., eps=1e-8,
                 **kwargs):
        super().__init__()

        self.vt = vt
        self.Rt = Rt
        self.rt = rt
        self.priory = priory
        self.priorpi = priorpi
        self.priorz = priorz
        self.sig_min = sig_min
        self.k = k
        self.beta = beta
        self.alpha = alpha
        self.tau = tau
        self.eps = eps

    def forward_var(self, pi, z):
        weighted_z_sum = torch.matmul(pi.T, z)
        class_sum = pi.sum(dim=0, keepdim=True).T
        mu = weighted_z_sum / (class_sum + self.eps)
        mu_diff = mu.unsqueeze(0) - mu.unsqueeze(1)
        between_class_distances = torch.sum(mu_diff ** 2, dim=2).sum()
        return - torch.sqrt(between_class_distances)
    
    def forward_mutual_information(self, pi):
        # Marginal entropy over batch — what you need
        pi_bar = pi.mean(0)  # [K] empirical component usage
        H_marginal = -(pi_bar * (pi_bar + 1e-8).log()).sum()  # should be log(K) = max

        # Maximizing H_marginal forces uniform component usage across batch
        return -H_marginal  # minimize this


    def forward_entropy(self, pi):
        return -(pi * torch.log(pi)).sum(-1)

    def forward(self, x: Tensor, hard=False) -> Tensor:
        B = len(x)
        _t = torch.rand(B, device=x.device)
        t = _t.reshape(-1, *([1] * (x.ndim - 1)))

        if self.priory is not None:
            x1 = self.priory.sample((B,)).to(x.device)
        else:
            x1 = torch.randn_like(x)
        xt = (1 - t) * x + (self.sig_min + (1 - self.sig_min) * t) * x1
        ut = (1 - self.sig_min) * x1 - x

        logitst, ztidx = self.Rt.rsample(xt, _t)
        pit = softmax(logitst/self.beta, dim=-1)
        zt = self.rt.rsample(logitst, xt, _t)

        vt = self.vt(_t, xt, zt)
        fm_loss = (vt - ut).square().mean(-1)
        loss = fm_loss.mean()
        # reg_loss = - pit.var(0).mean() + self.forward_entropy(pit).mean() * 0.001
        reg_loss = self.forward_mutual_information(pit)
        # reg_loss = - pit.var(0).mean()
        beta1 = loss.detach() / (torch.abs(reg_loss).detach() + 1e-8)
        return loss + beta1 * self.alpha * reg_loss
