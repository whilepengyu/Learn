import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Optional
from typing import Union
from typing import Tuple
from typing import List
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import numpy as np
import torch
import math
import os

class Tools:
    def gather(self, consts: torch.Tensor, t: torch.Tensor):
        c= consts.gather(-1, t)
        return c.reshape(-1, 1, 1, 1)

class Swish(nn.Module):
    def forward(self, x):
        return x* torch.sigmoid(x)

class ResidualBolck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, n_groups: int= 16, dropout: float= 0.1):
        super().__init__()
        self.norm1= nn.GroupNorm(n_groups, in_channels)
        self.act1= Swish()
        self.conv1= nn.Conv2d(in_channels, out_channels, kernel_size= (3, 3), padding= (1, 1))
        self.norm2= nn.GroupNorm(n_groups, out_channels)
        self.act2= Swish()
        self.conv2= nn.Conv2d(out_channels, out_channels, kernel_size= (3, 3), padding= (1, 1))
        if in_channels!= out_channels:
            self.shortcut= nn.Conv2d(in_channels, out_channels, kernel_size= (1, 1))
        else:
            self.shortcut= nn.Identity()
        self.time_emb= nn.Linear(time_channels, out_channels)
        self.time_act= Swish()
        self.dropout= nn.Dropout(dropout)
    def forward(self, x: torch.Tensor, t:torch.Tensor):
        h= self.conv1(self.act1(self.norm1(x)))
        h+= self.time_emb(self.time_act(t))[:, :, None, None]
        h= self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h+ self.shortcut(x)
    
class AttentionBlock(nn.Module):
    def __init__(self, n_channels: int, n_heads: int= 1, k_dims: int= None, n_groups: int= 16):
        super().__init__()
        if k_dims is None:
            k_dims= n_channels
        self.norm= nn.GroupNorm(n_groups, n_channels)
        self.projection= nn.Linear(n_channels, n_heads* k_dims* 3)
        self.output= nn.Linear(n_heads* k_dims, n_channels)
        self.scale= k_dims** -0.5
        self.n_heads= n_heads
        self.k_dims= k_dims
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor]= None):
        _= t
        batch_size, n_channels, height, weight= x.shape
        x= x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        qkv= self.projection(x).view(batch_size, -1, self.n_heads, 3* self.k_dims)
        q, k, v= torch.chunk(qkv, 3, dim= -1)
        attn= torch.einsum('bihd,bjhd->bijh', q, k)* self.scale
        attn= attn.softmax(dim= 2)
        res= torch.einsum('bijh,bjhd->bihd', attn, v)
        res= res.view(batch_size, -1, self.n_heads* self.k_dims)
        res= self.output(res)
        res+= x
        res= res.permute(0, 2, 1).view(batch_size, n_channels, height, weight)
        return res

class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        self.res= ResidualBolck(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn= AttentionBlock(out_channels)
        else:
            self.attn= nn.Identity()
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x= self.res(x, t)
        x= self.attn(x)
        return x

class TimeEmbedding(nn.Module):
    def __init__(self, n_channels:int):
        super().__init__()
        self.n_channels= n_channels
        self.lin1= nn.Linear(self.n_channels// 4, self.n_channels)
        self.act= Swish()
        self.lin2= nn.Linear(self.n_channels, self.n_channels)
    def forward(self, t: torch.Tensor):
        half_dim= self.n_channels// 8
        emb= math.log(10000)/ (half_dim- 1)
        emb= torch.exp(torch.arange(half_dim, device= t.device)* -emb)
        emb= t[:, None]* emb[None, :]
        emb= torch.cat((emb.sin(), emb.cos()), dim= 1)
        # transform
        emb= self.act(self.lin1(emb))
        emb= self.lin2(emb)
        return emb  

class Upsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv= nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        _= t
        return self.conv(x)

class Downsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv= nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))
    def forward(self, x:torch.tensor, t:torch.tensor):
        _= t
        return self.conv(x)

class MiddleBlock(nn.Module):
    def __init__(self, n_channels:int, time_channels:int):
        super().__init__()
        self.res1= ResidualBolck(n_channels, n_channels, time_channels)
        self.attn= AttentionBlock(n_channels)
        self.res2= ResidualBolck(n_channels, n_channels, time_channels)
    def forward(self, x:torch.tensor, t:torch.tensor):
        x= self.res1(x, t)
        x= self.attn(x)
        x= self.res2(x, t)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        self.res= ResidualBolck(in_channels+ out_channels, out_channels, time_channels)
        if has_attn:
            self.attn= AttentionBlock(out_channels)
        else:
            self.attn= nn.Identity()
    def forward(self, x:torch.Tensor, t:torch.Tensor):
        x= self.res(x, t)
        x= self.attn(x)
        return x

class UNet(nn.Module):
    def __init__(self, image_channels: int = 3, n_channels: int = 64,
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
                 is_attn: Union[Tuple[bool, ...], List[int]] = (False, False, True, True),
                 n_blocks: int = 2):
        super().__init__()
        n_resolutions = len(ch_mults)
        self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))
        self.time_emb = TimeEmbedding(n_channels * 4)
        down = []
        out_channels = in_channels = n_channels
        for i in range(n_resolutions):
            out_channels = in_channels * ch_mults[i]
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
                in_channels = out_channels
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        self.down = nn.ModuleList(down)

        self.middle = MiddleBlock(out_channels, n_channels * 4, )
        
        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
        
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            in_channels = out_channels
            
            if i > 0:
                up.append(Upsample(in_channels))

        self.up = nn.ModuleList(up)

        self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.final = nn.Conv2d(in_channels, image_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t = self.time_emb(t)
        x = self.image_proj(x)
        h = [x]
        for m in self.down:
            x = m(x, t)
            h.append(x)

        x = self.middle(x, t)
        
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, t)

        return self.final(self.act(self.norm(x)))
    
class DiffusionModel:
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        super().__init__()
        self.eps_model= eps_model
        self.beta= torch.linspace(0.0001, 0.02, n_steps).to(device)
        self.alpha= 1.0- self.beta
        self.alpha_bar= torch.cumprod(self.alpha, dim= 0)
        self.n_steps= n_steps
        self.sigma= self.beta
        self.tools= Tools()

    def q_xt_x0(self, x0: torch.tensor, t:torch.Tensor):
        mean= self.tools.gather(self.alpha_bar, t)** 0.5* x0
        var= 1- self.tools.gather(self.alpha_bar, t)
        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor]= None):
        if eps is None:
            eps= torch.randn_like(x0)
        mean, var= self.q_xt_x0(x0, t)
        return mean+ (var** 0.5)* eps
    
    def p_sample(self, xt: torch.tensor, t: torch.Tensor):
        eps_hat= self.eps_model(xt, t)
        alpha_bar= self.tools.gather(self.alpha_bar, t)
        alpha= self.tools.gather(self.alpha, t)
        eps_coef= (1- alpha)/ (1- alpha_bar)** 0.5
        mean= 1/ (alpha** 0.5)* (xt- eps_coef* eps_hat)
        var= self.tools.gather(self.sigma, t)
        eps= torch.randn(xt.shape, device= xt.device)
        return mean+ (var** 0.5)* eps
    
    def loss(self, x0: torch.tensor, noise: Optional[torch.Tensor]= None):
        batch_size= x0.shape[0]
        t= torch.randint(0, self.n_steps, (batch_size, ), device= x0.device, dtype= torch.long)
        if noise is None:noise= torch.randn_like(x0)
        xt= self.q_sample(x0, t, eps= noise)
        eps_hat= self.eps_model(xt, t)
        return F.mse_loss(noise, eps_hat)


def show_sample(images, texts):
    _, figs= plt.subplots(1, len(images), figsize= (12, 12))
    for text, f, img in zip(texts, figs, images):
        f.imshow(img.view(28, 28), cmap= 'gray')
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
        f.text(0.5, 0, text, ha= 'center', va= 'bottom', fontsize= 12, color= 'white', backgroundcolor= 'black')
    plt.show()

save_dir = "./u_net.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    seed, batch_size= 1, 128
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    
    model = UNet(1, 16, [1, 2, 2], [False, False, False], n_blocks= 1).to(device)
    diffusion = DiffusionModel(model, n_steps=1000, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_score, score, epochs, early_stop_time, early_stop_threshold= 1e10, 0, 100, 0, 20

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        loss_record= []
        for step, (pic, labels) in enumerate(train_loader):
            pic= pic.view(-1, 1, 28, 28).to(device)
            optimizer.zero_grad()
            loss= diffusion.loss(pic)
            loss_record.append(loss.item())
            loss.backward()
            optimizer.step()
        print(f'training epoch: {epoch}, mean loss: {torch.tensor(loss_record).mean()}')
        loss_record= []
        with torch.no_grad():
            for step, (pic, labels) in enumerate(valid_loader):
                pic= pic.view(-1, 1, 28, 28).to(device)
                loss= diffusion.loss(pic)
                loss_record.append(loss.item()) 
        mean_loss= torch.tensor(loss_record).mean()
        if mean_loss < best_score:
            early_stop_time= 0
            best_score= mean_loss
            torch.save(model, f'{save_dir}')
        else:
            early_stop_time= early_stop_time+ 1
        if early_stop_time> early_stop_threshold:
            break
        print(f'early_stop_time/early_stop_threshold: {early_stop_time}/{early_stop_threshold}, mean loss: {mean_loss}')

    
    
def inference():
    xt, images, texts= torch.randn((1, 1, 28, 28), device= device), [], []
    model= torch.load(f'{save_dir}')
    diffusion = DiffusionModel(model, n_steps=1000, device=device)
    for t in reversed(range(1000)):
        xt_1= diffusion.p_sample(xt, torch.tensor([t]).to(device))
        xt= xt_1
        if (t+ 1)% 100== 1:
            images.append(xt.view(1, 28, 28).to('cpu').detach())
            texts.append(t+ 1)
            
    images_= torch.stack(images, dim= 0)
    show_sample(images_, texts)
    
# train()
inference()