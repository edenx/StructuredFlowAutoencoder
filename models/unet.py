from abc import abstractmethod
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import grad, vmap
from functools import partial

torch.set_default_dtype(torch.float32)

# def timestep_embedding(timesteps:torch.Tensor, dim:int, freqs:int=2) -> torch.Tensor:
#     """
#     Create sinusoidal timestep embeddings.

#     :param timesteps: a 1-D Tensor of N indices, one per batch element.
#                       These may be fractional.
#     :param dim: the dimension of the output.
#     :param max_period: controls the minimum frequency of the embeddings.
#     :return: an [N x dim] Tensor of positional embeddings.
#     """
#     if timesteps.dim() == 0:
#         timesteps = timesteps.view(1)
#     # half = dim // 2
#     # freqs = torch.exp(
#     #     -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
#     # ).to(device=timesteps.device)
#     args = timesteps[:, None].float() * (freqs//2)
#     embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
#     # if dim % 2:
#     #     embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
#     return embedding

def timestep_embedding(timesteps: torch.Tensor, dim: int, freqs: int = 2) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings with an inputted frequency.

    :param timesteps: a 1-D Tensor of N timesteps, one per batch element. These may be fractional.
    :param dim: the dimension of the output.
    :param freqs: the number of frequency bands used for the embedding.
    :return: an [N x dim] Tensor of sinusoidal embeddings.
    """
    if timesteps.dim() == 0:
        timesteps = timesteps.view(1)

    # Half dimension for sin and cos parts
    half_dim = dim // 2

    # Create a frequency scale for the interval [0, 1]
    freq_scale = torch.linspace(1, freqs, half_dim, dtype=torch.float32, device=timesteps.device) * torch.pi
    
    # Scale timesteps by the frequency scale and expand to match the desired dimension
    scaled_timesteps = timesteps[:, None] * freq_scale[None, :]

    # Apply sine and cosine to the scaled timesteps
    embedding = torch.cat([torch.sin(scaled_timesteps), torch.cos(scaled_timesteps)], dim=-1)

    # If the dimension is odd, we need to pad the output to reach the specified dim size
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    return embedding

class Upsample(nn.Module):
    """
    an upsampling layer
    """
    def __init__(self, in_ch:int, out_ch:int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.layer = nn.Conv2d(in_ch, out_ch, kernel_size = 3, stride = 1, padding = 1)
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.in_ch, f'x and upsampling layer({self.in_ch}->{self.out_ch}) doesn\'t match.'
        x = F.interpolate(x, scale_factor = 2, mode = "nearest")
        output = self.layer(x)
        return output

class Downsample(nn.Module):
    """
    a downsampling layer
    """
    def __init__(self, in_ch:int, out_ch:int, use_conv:bool):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        if use_conv:
            self.layer = nn.Conv2d(self.in_ch, self.out_ch, kernel_size = 3, stride = 2, padding = 1)
        else:
            self.layer = nn.AvgPool2d(kernel_size = 2, stride = 2)
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.in_ch, f'x and upsampling layer({self.in_ch}->{self.out_ch}) doesn\'t match.'
        return self.layer(x)

class EmbedBlock(nn.Module):
    """
    abstract class
    """
    @abstractmethod
    def forward(self, x, temb, cemb):
        """
        abstract method
        """

class EmbedBlock_nz(nn.Module):
    """
    abstract class
    """
    @abstractmethod
    def forward(self, x, temb):
        """
        abstract method
        """

class EmbedSequential(nn.Sequential, EmbedBlock):
    def forward(self, x:torch.Tensor, temb:torch.Tensor, cemb:torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, EmbedBlock):
                x = layer(x, temb, cemb)
            else:
                x = layer(x)
        return x

class EmbedSequential_nz(nn.Sequential):
    def forward(self, x:torch.Tensor, temb:torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, EmbedBlock_nz):
                x = layer(x, temb)
            else:
                x = layer(x)
        return x


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, n_channels: int):
        super().__init__()
        self.film = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, n_channels * 2)  # outputs both gamma and beta
        )

    def forward(self, x, cond):
        gamma_beta = self.film(cond)  # [B, 2C]
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)  # Each is [B, C]

        # Reshape for broadcasting over H, W
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)    # [B, C, 1, 1]

        return gamma * x + beta


class ResBlock(EmbedBlock):
    def __init__(self, in_ch: int, out_ch: int, tdim: int, cdim: int, droprate: float):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.tdim = tdim
        self.cdim = cdim
        self.droprate = droprate
        cond_dim = tdim + cdim

        self.norm1 = nn.GroupNorm(8, in_ch)
        self.film1 = FiLM(cond_dim, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.film2 = FiLM(cond_dim, out_ch)
        self.dropout = nn.Dropout(p=droprate)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.residual = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor, cemb: torch.Tensor) -> torch.Tensor:

        cond = torch.cat([temb, cemb], dim=1)  # [B, cond_dim]

        out = self.norm1(x)
        out = self.film1(out, cond)
        out = F.silu(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.film2(out, cond)
        out = F.silu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        return out + self.residual(x)


class ResBlock_nz(EmbedBlock_nz):
    def __init__(self, in_ch:torch.Tensor, out_ch:torch.Tensor, tdim, droprate:float):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.tdim = tdim
        self.droprate = droprate

        self.block_1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            # nn.Softplus(),
            nn.Conv2d(in_ch, out_ch, kernel_size = 3, padding = 1),
        )

        self.temb_proj = nn.Sequential(
            nn.SiLU(),
            # nn.Softplus(),
            nn.Linear(tdim, out_ch),
        )
        
        self.block_2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            # nn.Softplus(),
            nn.Dropout(p = self.droprate),
            nn.Conv2d(out_ch, out_ch, kernel_size = 3, stride = 1, padding = 1),
            
        )
        if in_ch != out_ch:
            self.residual = nn.Conv2d(in_ch, out_ch, kernel_size = 1, stride = 1, padding = 0)
        else:
            self.residual = nn.Identity()
    
    def forward(self, x:torch.Tensor, temb) -> torch.Tensor:
        latent = self.block_1(x)
        latent += self.temb_proj(temb)[:, :, None, None]
        latent = self.block_2(latent)

        latent += self.residual(x)
        return latent


class ResBlock_nzt(nn.Module):
    def __init__(self, in_ch:torch.Tensor, out_ch:torch.Tensor, droprate:float):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        # self.tdim = tdim
        self.droprate = droprate

        self.block_1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            # nn.Softplus(),
            nn.Conv2d(in_ch, out_ch, kernel_size = 3, padding = 1),
        )

        # self.temb_proj = nn.Sequential(
        #     nn.SiLU(),
        #     # nn.Softplus(),
        #     nn.Linear(tdim, out_ch),
        # )
        
        self.block_2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            # nn.Softplus(),
            nn.Dropout(p = self.droprate),
            nn.Conv2d(out_ch, out_ch, kernel_size = 3, stride = 1, padding = 1),
            
        )
        if in_ch != out_ch:
            self.residual = nn.Conv2d(in_ch, out_ch, kernel_size = 1, stride = 1, padding = 0)
        else:
            self.residual = nn.Identity()
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        latent = self.block_1(x)
        # latent += self.temb_proj(temb)[:, :, None, None]
        latent = self.block_2(latent)

        latent += self.residual(x)
        return latent
        

class AttnBlock(nn.Module):
    def __init__(self, in_ch:int):
        super().__init__()
        self.group_norm = nn.GroupNorm(8, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, kernel_size = 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, kernel_size = 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, kernel_size = 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, kernel_size = 1, stride=1, padding=0)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class Unet(nn.Module):
    def __init__(self, in_ch=3, mod_ch=64, out_ch=3, ch_mul=[1,2,4,8], num_res_blocks=2, cdim=10, use_conv=True, droprate=0, freqs=2, dtype=torch.float32):
        super().__init__()
        self.in_ch = in_ch
        self.mod_ch = mod_ch
        self.out_ch = out_ch
        self.ch_mul = ch_mul
        self.num_res_blocks = num_res_blocks
        self.cdim = cdim
        self.use_conv = use_conv
        self.droprate = droprate
        # self.num_heads = num_heads
        self.freqs = freqs
        self.dtype = dtype
        tdim = 2 * freqs
        _cdim = mod_ch * 4

        self.cemb_layer = nn.Linear(cdim, _cdim)

        self.downblocks = nn.ModuleList([
            EmbedSequential(nn.Conv2d(in_ch, self.mod_ch, 3, padding=1))
        ])
        now_ch = self.ch_mul[0] * self.mod_ch
        
        chs = [now_ch]
        for i, mul in enumerate(self.ch_mul):
            nxt_ch = mul * self.mod_ch
            
            for _ in range(self.num_res_blocks):
                layers = [
                    ResBlock(now_ch, nxt_ch, tdim, _cdim, self.droprate),
                    AttnBlock(nxt_ch)
                ]
                now_ch = nxt_ch
                self.downblocks.append(EmbedSequential(*layers))
                chs.append(now_ch)
            if i != len(self.ch_mul) - 1:
                self.downblocks.append(EmbedSequential(Downsample(now_ch, now_ch, self.use_conv)))
                chs.append(now_ch)
        self.middleblocks = EmbedSequential(
            ResBlock(now_ch, now_ch, tdim, _cdim, self.droprate),
            AttnBlock(now_ch),
            ResBlock(now_ch, now_ch, tdim, _cdim, self.droprate)
        )
        self.upblocks = nn.ModuleList([])
        for i, mul in list(enumerate(self.ch_mul))[::-1]:
            nxt_ch = mul * self.mod_ch
            for j in range(num_res_blocks + 1):
                layers = [
                    ResBlock(now_ch+chs.pop(), nxt_ch, tdim, _cdim, self.droprate),
                    AttnBlock(nxt_ch)
                ]
                now_ch = nxt_ch
                if i and j == self.num_res_blocks:
                    layers.append(Upsample(now_ch, now_ch))
                self.upblocks.append(EmbedSequential(*layers))
        
        self.out = nn.Sequential(
            nn.GroupNorm(8, now_ch),
            nn.SiLU(),
            # nn.Softplus(),
            nn.Conv2d(now_ch, self.out_ch, 3, stride = 1, padding = 1),
            # nn.Tanh()
        )
    def forward(self, t:torch.Tensor, x:torch.Tensor, z:torch.Tensor) -> torch.Tensor:
        temb = timestep_embedding(t, self.freqs*2, self.freqs).expand(len(x),-1)
        # print("temb", temb.shape)
        # temb = self.temb_layer(temb).expand(len(x),-1)
        cemb = self.cemb_layer(z)
        # print("t", temb.shape)
        # print("cemb", cemb.shape)
        
        hs = []
        h = x.type(self.dtype)
        for block in self.downblocks:
            h = block(h, temb, cemb)
            hs.append(h)
        h = self.middleblocks(h, temb, cemb)

        for block in self.upblocks:
            _hs = hs.pop()
            if _hs.shape[-2:] != h.shape[-2:]:
                h = F.interpolate(h, size=_hs.shape[-2:], mode='nearest')  # Upsample _hs to match h
            h = torch.cat([h, _hs], dim = 1)
            h = block(h, temb, cemb)
        h = h.type(self.dtype)
        return self.out(h)

class Unet_nz(nn.Module):
    def __init__(self, in_ch=3, mod_ch=64, out_ch=3, ch_mul=[1,2,4,8], num_res_blocks=2, cdim=10, use_conv=True, droprate=0, freqs=2, dtype=torch.float32):
        super().__init__()
        self.in_ch = in_ch
        self.mod_ch = mod_ch
        self.out_ch = out_ch
        self.ch_mul = ch_mul
        self.num_res_blocks = num_res_blocks
        self.cdim = cdim
        self.use_conv = use_conv
        self.droprate = droprate
        # self.num_heads = num_heads
        self.freqs = freqs
        self.dtype = dtype
        tdim = mod_ch * 4
        self.temb_layer = nn.Sequential(
            nn.Linear(mod_ch, tdim),
            nn.SiLU(),
            # nn.Softplus(),
            nn.Linear(tdim, tdim),
        )
        self.downblocks = nn.ModuleList([
            EmbedSequential_nz(nn.Conv2d(in_ch, self.mod_ch, 3, padding=1))
        ])
        now_ch = self.ch_mul[0] * self.mod_ch
        
        chs = [now_ch]
        for i, mul in enumerate(self.ch_mul):
            nxt_ch = mul * self.mod_ch
            
            for _ in range(self.num_res_blocks):
                layers = [
                    ResBlock_nz(now_ch, nxt_ch, tdim, self.droprate),
                    AttnBlock(nxt_ch)
                ]
                now_ch = nxt_ch
                self.downblocks.append(EmbedSequential_nz(*layers))
                chs.append(now_ch)
            if i != len(self.ch_mul) - 1:
                self.downblocks.append(EmbedSequential_nz(Downsample(now_ch, now_ch, self.use_conv)))
                chs.append(now_ch)
        self.middleblocks = EmbedSequential_nz(
            ResBlock_nz(now_ch, now_ch, tdim, self.droprate),
            AttnBlock(now_ch),
            ResBlock_nz(now_ch, now_ch, tdim, self.droprate)
        )
        self.upblocks = nn.ModuleList([])
        for i, mul in list(enumerate(self.ch_mul))[::-1]:
            nxt_ch = mul * self.mod_ch
            for j in range(num_res_blocks + 1):
                layers = [
                    ResBlock_nz(now_ch+chs.pop(), nxt_ch, tdim, self.droprate),
                    AttnBlock(nxt_ch)
                ]
                now_ch = nxt_ch
                if i and j == self.num_res_blocks:
                    layers.append(Upsample(now_ch, now_ch))
                self.upblocks.append(EmbedSequential_nz(*layers))
        
        self.out = nn.Sequential(
            nn.GroupNorm(8, now_ch),
            nn.SiLU(),
            # nn.Softplus(),
            nn.Conv2d(now_ch, self.out_ch, 3, stride = 1, padding = 1),
            # nn.Tanh()
        )
    def forward(self, t:torch.Tensor, x:torch.Tensor) -> torch.Tensor:
        temb = timestep_embedding(t, self.mod_ch, self.freqs)
        # print("temb", temb.shape)
        temb = self.temb_layer(temb)
        hs = []
        h = x.type(self.dtype)
        for block in self.downblocks:
            h = block(h, temb)
            hs.append(h)
        h = self.middleblocks(h, temb)

        for block in self.upblocks:
            _hs = hs.pop()
            if _hs.shape[-2:] != h.shape[-2:]:
                h = F.interpolate(h, size=_hs.shape[-2:], mode='nearest')  # Upsample _hs to match h
            h = torch.cat([h, _hs], dim = 1)
            h = block(h, temb)
        h = h.type(self.dtype)
        return self.out(h)


class Unet_nzt(nn.Module):
    def __init__(self, in_ch=3, mod_ch=64, out_ch=3, ch_mul=[1,2,4,8], num_res_blocks=2, use_conv=True, droprate=0, dtype=torch.float32):
        super().__init__()
        self.in_ch = in_ch
        self.mod_ch = mod_ch
        self.out_ch = out_ch
        self.ch_mul = ch_mul
        self.num_res_blocks = num_res_blocks
        # self.cdim = cdim
        self.use_conv = use_conv
        self.droprate = droprate
        # self.num_heads = num_heads
        # self.freqs = freqs
        self.dtype = dtype
        # tdim = mod_ch * 4
        # self.temb_layer = nn.Sequential(
        #     nn.Linear(mod_ch, tdim),
        #     nn.SiLU(),
        #     # nn.Softplus(),
        #     nn.Linear(tdim, tdim),
        # )
        self.downblocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_ch, self.mod_ch, 3, padding=1))
        ])
        now_ch = self.ch_mul[0] * self.mod_ch
        
        chs = [now_ch]
        for i, mul in enumerate(self.ch_mul):
            nxt_ch = mul * self.mod_ch
            
            for _ in range(self.num_res_blocks):
                layers = [
                    ResBlock_nzt(now_ch, nxt_ch, self.droprate),
                    AttnBlock(nxt_ch)
                ]
                now_ch = nxt_ch
                self.downblocks.append(nn.Sequential(*layers))
                chs.append(now_ch)
            if i != len(self.ch_mul) - 1:
                self.downblocks.append(nn.Sequential(Downsample(now_ch, now_ch, self.use_conv)))
                chs.append(now_ch)
        self.middleblocks = nn.Sequential(
            ResBlock_nzt(now_ch, now_ch, self.droprate),
            AttnBlock(now_ch),
            ResBlock_nzt(now_ch, now_ch, self.droprate)
        )
        self.upblocks = nn.ModuleList([])
        for i, mul in list(enumerate(self.ch_mul))[::-1]:
            nxt_ch = mul * self.mod_ch
            for j in range(num_res_blocks + 1):
                layers = [
                    ResBlock_nzt(now_ch+chs.pop(), nxt_ch, self.droprate),
                    AttnBlock(nxt_ch)
                ]
                now_ch = nxt_ch
                if i and j == self.num_res_blocks:
                    layers.append(Upsample(now_ch, now_ch))
                self.upblocks.append(nn.Sequential(*layers))
        
        self.out = nn.Sequential(
            nn.GroupNorm(8, now_ch),
            nn.SiLU(),
            # nn.Softplus(),
            nn.Conv2d(now_ch, self.out_ch, 3, stride = 1, padding = 1),
            # nn.Tanh()
        )
    def forward(self, x:torch.Tensor) -> torch.Tensor:

        hs = []
        h = x.type(self.dtype)
        for block in self.downblocks:
            h = block(h)
            hs.append(h)
        h = self.middleblocks(h)

        for block in self.upblocks:
            _hs = hs.pop()
            if _hs.shape[-2:] != h.shape[-2:]:
                h = F.interpolate(h, size=_hs.shape[-2:], mode='nearest')  # Upsample _hs to match h
            h = torch.cat([h, _hs], dim = 1)
            h = block(h)
        h = h.type(self.dtype)
        return self.out(h)

