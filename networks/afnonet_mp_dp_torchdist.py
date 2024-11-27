#reference: https://github.com/NVlabs/AFNO-transformer

import math
from functools import partial
from collections import OrderedDict
from copy import Error, deepcopy
from re import S
import time
from numpy.lib.arraypad import pad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
import torch.fft
from torch.nn.modules.container import Sequential
from torch.utils.checkpoint import checkpoint_sequential
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from utils.img_utils import PeriodicPad2d
from mpi4py import MPI
import torch.distributed as dist
from utils.comm import init_local_group
import logging
DEBUG = True
FLAG = 0
#TODO:
## DONE #functions for communication, forward and backward pass MLP and AFNO2D
## DONE #Split model size by world size MLP and AFNO2D
## DONE #what is to be split and what is to be communicated
#Dropout seeds plus epoch oder so -> über torch.randomseed oder test ob MLP dropout 1 benötigt

# create comm group depending on world size and mp parallel size
mp_group = None
dp_group = None
compute_time = 0
comm_time = 0
reformat_time = 0

def create_dp_group(mp_size):
    global dp_group
    ranks = []
    for i in range(MPI.COMM_WORLD.Get_size()):
        if i%mp_size == 0:
            ranks.append(i)
    dp_group = dist.new_group(ranks=ranks,backend="nccl", group_desc="dp_group")### Added by Robin Maurer

def create_mp_group(mp_size):
    global mp_group
    rank = dist.get_rank()
    #if rank % mp_size == 0:
    mp_group = dist.new_group(ranks=[i for i in range(rank-rank%mp_size, rank-rank%mp_size+mp_size)],backend="nccl", group_desc="mp_group")



def allreduceWrapper(input:torch.Tensor, mp_size):
    dp_size= dist.get_world_size()//mp_size
    dist.all_reduce(input, op=dist.ReduceOp.SUM,group=dp_group)
    ret = torch.div(input,dp_size)
    return ret

def broadcastWrapper(input, mp_size):
    if mp_size > 1:
        rank = dist.get_rank()
        src = rank - rank % mp_size
        dist.broadcast(input, src=src, group=mp_group)
    return input

def scatterWrapper(input, mp_size, rank):
    global comm_time, reformat_time                     ## Time measurement
    re_start_time = time.time()                         ## Time measurement
    rearranged_input = input.permute(3,0,1,2)
    rearranged_input = rearranged_input.contiguous()
    reformat_time += time.time() - re_start_time        ## Time measurement
    src = rank - rank % mp_size
    local_rank = rank % mp_size
    
    split_tensors = list(torch.chunk(rearranged_input, mp_size, dim=0))
    received_tensor = torch.zeros(split_tensors[local_rank].shape,dtype=input.dtype,device=input.device,requires_grad=True)
    comm_start_time = time.time()                       ## Time measurement
    dist.scatter(received_tensor,split_tensors if src == rank else None, src=src,group=mp_group)
    comm_time += time.time() - comm_start_time          ## Time measurement
    re_start_time = time.time()                         ## Time measurement
    received_tensor_orig_shape = received_tensor.permute(1,2,3,0)
    received_tensor_orig_shape = received_tensor_orig_shape.contiguous()
    reformat_time += time.time() - re_start_time        ## Time measurement
    return received_tensor_orig_shape

def gatherWrapper(input, mp_size, rank, device):
    global comm_time, reformat_time
    re_start_time = time.time()                         ## Time measurement
    rearranged_input = input.permute(3,0,1,2)
    rearranged_input = rearranged_input.contiguous()
    reformat_time += time.time() - re_start_time        ## Time measurement
    gathered_tensor_list = [
        torch.zeros(rearranged_input.shape,dtype=input.dtype, device=input.device, requires_grad=True)
        for _ in range(mp_size)]
    comm_start_time = time.time()                       ## Time measurement
    dist.all_gather(gathered_tensor_list, rearranged_input, group=mp_group)
    comm_time += time.time() - comm_start_time          ## Time measurement
    gathered_tensor = [t.to(device) for t in gathered_tensor_list]            
    received_tensor = torch.cat(gathered_tensor, dim=0) if gathered_tensor is not None else None
    re_start_time = time.time()                         ## Time measurement
    gathered_tensor_orig_shape = received_tensor.permute(1,2,3,0).contiguous()
    reformat_time += time.time() - re_start_time        ## Time measurement
    return gathered_tensor_orig_shape


class AllReduceBcast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mp_size):
        ctx.mp_size = mp_size
        return input
        if dist.get_world_size() == 1:
            return input
        input = allreduceWrapper(input, mp_size)
        return input
    
    def backward(ctx, grad_output):
        if dist.get_world_size() == 1:
            return grad_output
        grad_output = broadcastWrapper(grad_output,ctx.mp_size)
        return grad_output, None, None, None
        
    

class ScatterGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mp_size, rank):
        ctx.mp_size = mp_size
        if mp_size == 1:
            return input
        if DEBUG:
            if rank == 0 and FLAG < 2:
                print("ScatterGather forward","Rank:", rank, "Input shape forward:", input.shape, "input device:", input.device, "input size in bytes:", 4*input.nelement(),"mp_size:",mp_size)
        ctx.device = input.device
        ctx.rank = rank
        
        received_tensor = scatterWrapper(input, mp_size, rank)
        
        return received_tensor

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.mp_size == 1:
            return grad_output, None, None, None
        global FLAG
        if DEBUG:
            if ctx.rank == 0 and FLAG < 2:
                FLAG += 1
                print("ScatterGather backward","Rank:", ctx.rank, "Input shape forward:", grad_output.shape, "input device:", grad_output.device, "input size in bytes:", 4*grad_output.nelement(),"mp_size:",ctx.mp_size)

        device = ctx.device
        grad_output = gatherWrapper(grad_output, ctx.mp_size, ctx.rank, device)
        
        return grad_output, None, None, None
    
class GatherScatter(torch.autograd.Function):
    @staticmethod
    # def forward(ctx, input, mp_size, rank, mp_group:MPI.Intracomm=None):
    def forward(ctx, input, mp_size, rank):
        ctx.mp_size = mp_size
        if mp_size == 1:
            return input
        if DEBUG:
            if rank == 0 and FLAG < 2:
                print("GatherScatter forward","Rank:", rank, "Input shape forward:", input.shape, "input device:", input.device, "input size in bytes:", 4*input.nelement(),"mp_size:",mp_size)

        ctx.device=input.device
        ctx.rank = rank
        device = input.device
        
        input = gatherWrapper(input, mp_size, rank, device)

        return input
        
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.mp_size == 1:
            return grad_output, None, None, None
        if DEBUG:
            if ctx.rank == 0 and FLAG < 2:
                print("GatherScatter backward","Rank:", ctx.rank, "Input shape forward:", grad_output.shape, "input device:", grad_output.device, "input size in bytes:", 4*grad_output.nelement(),"mp_size:",ctx.mp_size)

        src = ctx.rank - ctx.rank % ctx.mp_size
        
        grad_output = scatterWrapper(grad_output, ctx.mp_size, ctx.rank)
        return grad_output, None, None, None
        # return ctx.mp_group.scatter(grad_output, ctx.rank - ctx.rank % ctx.mp_size), None, None, None

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # hidden_features_local = hidden_features // get_world_size()
        self.fc1 = nn.Linear(in_features, hidden_features)#_local)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        # self.fc2 = nn.Linear(hidden_features_local, out_features)
        self.drop = nn.Dropout(drop)
        # self.gather_shapes = compute_split_shapes(
        #         in_features*get_world_size(), get_world_size()
        #     )

    def forward(self, x):
        #x = gather_from_parallel_region(x, dim=1, shapes=self.gather_shapes,group=None)
        # x = copy_to_parallel_region(x, group=None)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        # x = reduce_from_parallel_region(x, group=None)
        x = self.drop(x)
        #x = scatter_to_parallel_region(x, dim=1, group=None)
        return x


class AFNO2D(nn.Module):
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1):
        super().__init__()
        assert hidden_size % (num_blocks) == 0, f"hidden_size {hidden_size} should be divisble by {num_blocks} (num_blocks {num_blocks})"
        
        self.hidden_size = hidden_size #768 
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks #8
        self.block_size = (self.hidden_size // self.num_blocks)  #96/world_size
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        # w1.shape = 2, 8, 96/world_size, 768; w2.shape = 2, 8, 768, 96/world_size; b1.shape = 2, 8, 768; b2.shape = 2, 8, 96/world_size
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, H, W, C = x.shape #2, 720, 1440, 768 = 2*720*1440*768 = 50960793600

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size) #2, 720, 721, 8, 96/world_size = 2*720*721*8*96/world_size = 25515786240

        o1_real = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)


        total_modes = H // 2 + 1 #721
        kept_modes = int(total_modes * self.hard_thresholding_fraction) #721

        o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        ) #2, 720, 721, 8, 96/world_size x 8, 96/world_size, 768 = 2, 720, 721, 8, 768

        o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        ) #2, 720, 721, 8, 96/world_size x 8, 96/world_size, 768 = 2, 720, 721, 8, 768

        o2_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes]  = (
            torch.einsum('...bi,bio->...bo', o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[0]
        ) #2, 720, 721, 8, 768 x 8, 768, 96/world_size = 2, 720, 721, 8, 96/world_size

        o2_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes]  = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[1]
        ) #2, 720, 721, 8, 768 x 8, 768, 96/world_size = 2, 720, 721, 8, 96/world_size

        x = torch.stack([o2_real, o2_imag], dim=-1) #2, 720, 721, 8, 96/world_size, 2
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x) #2, 720, 721, 8, 96/world_size
        x = x.reshape(B, H, W // 2 + 1, C) # 2, 720, 721, 768
        x = torch.fft.irfft2(x, s=(H, W), dim=(1,2), norm="ortho") #2, 720, 1440, 768
        x = x.type(dtype)

        return x + bias


class Block(nn.Module):
    def __init__(
            self,
            dim,
            mlp_ratio=4.,
            drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            double_skip=True,
            num_blocks=8,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
            input_parallel=False,
            output_parallel=False,
            mp_size=1,
        ):
        super().__init__()
        dim = dim
        self.input_parallel = input_parallel
        self.output_parallel = output_parallel
        self.norm1 = norm_layer(dim)
        self.filter = AFNO2D(dim, num_blocks, sparsity_threshold, hard_thresholding_fraction) 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.double_skip = double_skip
        self.scatter_fn = ScatterGather.apply
        self.gather_fn = GatherScatter.apply
        self.mp_size = mp_size
        

    def forward(self, x):
        global comm_time
        rank = MPI.COMM_WORLD.Get_rank()
        # mp_group = create_mp_comm_group(self.mp_size)
        if not self.input_parallel:
            #print(x.shape)
            # x = self.scatter_fn(x, self.mp_size, rank, mp_group)
            x = self.scatter_fn(x, self.mp_size, rank)
            # device = next(self.parameters()).device  # oder z.B. torch.device("cuda:0")
            # x = x.to(device)

            #print(x.shape)
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        if not self.output_parallel:
            #print(x.shape)
            # x = self.gather_fn(x, self.mp_size, rank, mp_group)
            x = self.gather_fn(x, self.mp_size, rank)
            #print(x.shape)
        return x

class PrecipNet(nn.Module):
    def __init__(self, params, backbone):
        super().__init__()
        self.params = params
        self.patch_size = (params.patch_size, params.patch_size)
        self.in_chans = params.N_in_channels
        self.out_chans = params.N_out_channels
        self.backbone = backbone
        self.ppad = PeriodicPad2d(1)
        self.conv = nn.Conv2d(self.out_chans, self.out_chans, kernel_size=3, stride=1, padding=0, bias=True)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.backbone(x)
        x = self.ppad(x)
        x = self.conv(x)
        x = self.act(x)
        return x

class AFNONetMPDP(nn.Module):
    def __init__(
            self,
            params,
            img_size=(720, 1440),
            patch_size=(16, 16),
            in_chans=2,
            out_chans=2,
            embed_dim=768,
            depth=12,
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.,
            num_blocks=16,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
        ):
        super().__init__()
        self.head_is_synced = False
        self.params = params
        mp_size = params.mp_size
        global mp_group
        mp_group = init_local_group(mp_size,1)
        self.compute_time = 0
        self.comm_time = 0
        self.reformat_time = 0
        #create_mp_group(mp_size)
        #create_dp_group(mp_size)
        self.img_size = img_size
        self.patch_size = (params.patch_size, params.patch_size)
        self.in_chans = params.N_in_channels
        self.out_chans = params.N_out_channels
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = params.num_blocks 
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.ddp_fn = AllReduceBcast.apply

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.h = img_size[0] // self.patch_size[0]
        self.w = img_size[1] // self.patch_size[1]

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim//mp_size, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
            num_blocks=self.num_blocks, sparsity_threshold=sparsity_threshold, hard_thresholding_fraction=hard_thresholding_fraction,
            input_parallel=False if i == 0 else True, output_parallel=True if i < depth-1 else False, mp_size=mp_size) 
        for i in range(depth)]) #NumBlocks = Diagonalmatrix von Attention

        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, self.out_chans*self.patch_size[0]*self.patch_size[1], bias=False)

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed # TODO: TEST Print pos_embed
        #if DEBUG: print("pose_embed",self.pos_embed)
        x = self.pos_drop(x)
        
        x = x.reshape(B, self.h, self.w, self.embed_dim)
        for blk in self.blocks:
            x = blk(x)

        return x

    def forward(self, x):
        global compute_time, comm_time, reformat_time
        compute_time = 0
        comm_time = 0
        reformat_time = 0
        start_time = time.time()
        x = self.forward_features(x)
        if not self.head_is_synced:
            self.head_is_synced = True
            start_comm_time = time.time()
            for param in self.head.parameters():
                dist.broadcast(
                    param, 0, group=None
                )
            comm_time += time.time() - start_comm_time
        x = self.head(x)
        x = rearrange(
            x,
            "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            h=self.img_size[0] // self.patch_size[0],
            w=self.img_size[1] // self.patch_size[1],
        )
        compute_time += time.time() - start_time
        # if MPI.COMM_WORLD.Get_size()>self.params.mp_size:
        #     x= self.ddp_fn(x, self.params.mp_size)
        global FLAG
        if FLAG == 0:
            print("Rank:", MPI.COMM_WORLD.Get_rank(),"Compute Time:", compute_time-comm_time,"Comm Time:", comm_time)
            FLAG = 1
        self.compute_time += compute_time-comm_time-reformat_time
        self.comm_time += comm_time
        self.reformat_time += reformat_time
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) # 14*14 = 196
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


if __name__ == "__main__":
    model = AFNONetMPDP(img_size=(720, 1440), patch_size=(4,4), in_chans=3, out_chans=10)
    sample = torch.randn(1, 3, 720, 1440)
    result = model(sample)
    print(result.shape)
    print(torch.norm(result))

