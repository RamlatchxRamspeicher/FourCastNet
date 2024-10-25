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
DEBUG = True
FLAG = 0
#TODO:
## DONE #functions for communication, forward and backward pass MLP and AFNO2D
## DONE #Split model size by world size MLP and AFNO2D
## DONE #what is to be split and what is to be communicated
#Dropout seeds plus epoch oder so -> über torch.randomseed oder test ob MLP dropout 1 benötigt

# create comm group depending on world size and mp parallel size
mp_group = None
compute_time = 0
comm_time = 0
def create_mp_comm_group(mp_size):
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()
    assert world_size % mp_size == 0, f"world_size {world_size} should be divisible by mp_size {mp_size}"
    # if rank % mp_size == 0:
    #     comm_group = comm.group.Incl([i for i in range(rank, rank+mp_size)])
    comm_group = comm.group.Incl([i for i in range(rank-rank%mp_size, rank-rank%mp_size+mp_size)])
    # print(f"Rank {rank} in group {comm_group.handle} has group size {comm_group.Get_size()} and group rank {comm_group.Get_rank()}")
    return comm.Create_group(comm_group)

# def get_mp_comm_group():
#     comm = MPI.COMM_WORLD
#     return comm.Get_group()

class ScatterGather(torch.autograd.Function):
    @staticmethod
    # def forward(ctx, input, mp_size, rank, mp_group:MPI.Intracomm=None):
    def forward(ctx, input, mp_size, rank):
        ctx.mp_size = mp_size
        if mp_size == 1:
            return input
        if DEBUG:
            if rank == 0 and FLAG < 2:
                print("ScatterGather forward","Rank:", rank, "Input shape forward:", input.shape, "input device:", input.device, "input size in bytes:", input.element_size()*input.nelement(),"mp_size:",mp_size)
        ctx.device =input.device
        ctx.rank = rank
        # ctx.mp_group = mp_group
        # print("len input forward:",len(input),"shape input forward:",input.shape)
        # scatter input along dim=3 to all ranks in group
        split_tensors = list(torch.chunk(input, mp_size, dim=3))
        if DEBUG and FLAG < 2:
            start_measure = time.time()
        if ctx.mp_size == 8:
            if ctx.rank%ctx.mp_size == 0:
                for i in range(1,ctx.mp_size):
                    mp_group.send(split_tensors[i], dest=i)
                    input = split_tensors[0]
            else:
                input = mp_group.recv(source=0)
        else:
            input = mp_group.scatter(split_tensors, root=0)
        if DEBUG and FLAG < 2:
            print("Scatter time forward for one batch:",time.time()-start_measure, "rank:",rank)
            start_measure = time.time()
        mp_group.barrier()
        if DEBUG and FLAG < 2:
            print("Barrier time scattergather forward for one batch:",time.time()-start_measure, "rank:",rank)
        # print("len scatter forward:",len(input),"shape scatter forward rank:",input[rank%mp_size].shape)
        return input
        # return mp_group.scatter(input, rank - rank % mp_size)
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.mp_size == 1:
            return grad_output, None, None, None
        global FLAG
        if DEBUG:
            if ctx.rank == 0 and FLAG < 2:
                FLAG += 1
                print("ScatterGather backward","Rank:", ctx.rank, "Input shape forward:", grad_output.shape, "input device:", grad_output.device, "input size in bytes:", grad_output.element_size()*grad_output.nelement(),"mp_size:",ctx.mp_size)

        device = ctx.device
        if DEBUG and FLAG < 2:
            start_measure = time.time()
        grad_output =  mp_group.allgather(grad_output)
        if DEBUG and FLAG < 2:
            print("scattergather time backward for one batch:",time.time()-start_measure, "rank:",ctx.rank)
            start_measure = time.time()
        # grad_output =  ctx.mp_group.allgather(grad_output)
        # ctx.mp_group.barrier()
        mp_group.barrier()
        if DEBUG and FLAG < 2:
            print("Barrier time scattergather backward for one batch:",time.time()-start_measure, "rank:",ctx.rank)
        grad_output = [t.to(device) for t in grad_output]
        # if grad_output is not None:
        #     print("len scatter backward:",len(grad_output),"shape scatter backward rank:", ctx.rank, ["tensor: " + str(i) + " device: " + str(grad_output[i].get_device()) for i in range(len(grad_output))])
        grad_output = torch.cat(grad_output, dim=3) if grad_output is not None else grad_output
        # print("len scatter backward:",len(grad_output),"shape scatter backward rank:",grad_output.shape)

        return grad_output, None, None, None
        # return ctx.mp_group.gather(grad_output, ctx.rank - ctx.rank % ctx.mp_size), None, None, None
    
class GatherScatter(torch.autograd.Function):
    @staticmethod
    # def forward(ctx, input, mp_size, rank, mp_group:MPI.Intracomm=None):
    def forward(ctx, input, mp_size, rank):
        ctx.mp_size = mp_size
        if mp_size == 1:
            return input
        if DEBUG:
            if rank == 0 and FLAG < 2:
                print("GatherScatter forward","Rank:", rank, "Input shape forward:", input.shape, "input device:", input.device, "input size in bytes:", input.element_size()*input.nelement(),"mp_size:",mp_size)

        ctx.rank = rank
        # ctx.mp_group = mp_group
        device = input.device
        ctx.device = device
        # print("device gather input forward:",device, "rank:",rank)
        # print("len  gather input forward:",len(input),"shape gather input forward:",input.shape)
        if DEBUG and FLAG < 2:
            start_measure = time.time()
        gathered_tensor = mp_group.allgather(input)
        if DEBUG and FLAG < 2:
            print("gatherscatter time forward for one batch:",time.time()-start_measure, "rank:",rank)
            start_measure = time.time()
        mp_group.barrier()
        if DEBUG and FLAG < 2:
            print("Barrier time gatherscatter forward for one batch:",time.time()-start_measure, "rank:",rank)
        # if gathered_tensor is not None:
        #     print("len gather forward:",len(gathered_tensor),"shape gather forward rank:", ctx.rank, ["tensor: " + str(i) + " device: " + str(gathered_tensor[i].get_device()) for i in range(len(gathered_tensor))])
        gathered_tensor = [t.to(device) for t in gathered_tensor]
        # for t in gathered_tensor:
        #     t = t.to(device)
        input = torch.cat(gathered_tensor, dim=3) if gathered_tensor is not None else gathered_tensor
        # print("len gather forward:",len(input),"shape gather forward rank:",input[rank%mp_size].shape)

        return input
        # return mp_group.gather(input, rank - rank % mp_size)
        
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.mp_size == 1:
            return grad_output, None, None, None
        if DEBUG:
            if ctx.rank == 0 and FLAG < 2:
                print("GatherScatter backward","Rank:", ctx.rank, "Input shape forward:", grad_output.shape, "input device:", grad_output.device, "input size in bytes:", grad_output.element_size()*grad_output.nelement(),"mp_size:",ctx.mp_size)

        device = ctx.device
        # print("len grad_output backward:",len(grad_output),"shape grad_output backward:",grad_output.shape)
        split_tensors = torch.chunk(grad_output, ctx.mp_size, dim=3)
        if DEBUG and FLAG < 2:
            start_measure = time.time()
        if ctx.mp_size == 8:
            if ctx.rank == 0:
                for i in range(1,ctx.mp_size):
                    mp_group.send(split_tensors[i], dest=i)
                    grad_output = split_tensors[0]
            else:
                grad_output = mp_group.recv(source=0)
        else:
            grad_output =  mp_group.scatter(split_tensors, root=0)
        if DEBUG and FLAG < 2:
            print("gatherscatter time backward for one batch:",time.time()-start_measure, "rank:",ctx.rank)
            start_measure = time.time()
        # grad_output =  ctx.mp_group.scatter(split_tensors, root=0)
        mp_group.barrier()
        if DEBUG and FLAG < 2:
            print("Barrier time gatherscatter backward for one batch:",time.time()-start_measure, "rank:",ctx.rank)
        # ctx.mp_group.barrier()
        # print("len gather backward:",len(grad_output),"shape gather backward rank:",grad_output[ctx.rank%ctx.mp_size].shape)

        return grad_output.to(device), None, None, None
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
            start_time = time.time()
            #print(x.shape)
            # x = self.scatter_fn(x, self.mp_size, rank, mp_group)
            x = self.scatter_fn(x, self.mp_size, rank)
            device = next(self.parameters()).device  # oder z.B. torch.device("cuda:0")
            x = x.to(device)
            comm_time += time.time() - start_time

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
            start_time = time.time()
            #print(x.shape)
            # x = self.gather_fn(x, self.mp_size, rank, mp_group)
            x = self.gather_fn(x, self.mp_size, rank)
            comm_time += time.time() - start_time
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
        mp_group = create_mp_comm_group(mp_size)
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
        global compute_time, comm_time
        start_time = time.time()
        x = self.forward_features(x)
        if not self.head_is_synced:
            start_comm_time = time.time()
            self.head_is_synced = True
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
        global FLAG
        if FLAG == 0:
            print("Rank:", MPI.COMM_WORLD.Get_rank(),"Compute Time:", compute_time,"Comm Time:", comm_time)
            FLAG = 1
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

