from timeit import timeit
import torch
import torch.distributed as dist
from mpi4py import MPI
from communicationHEAT import MPICommunication
from comm import init
import logging
import time
from functools import wraps




def main(mp_size=2):

    init("nccl-slurm")
    # check for cuda aware mpi

    comm = MPI.COMM_WORLD
    commHEAT = MPICommunication(comm)
    rank= MPI.COMM_WORLD.Get_rank()
    mp_size = mp_size
    start = time.time()
    tensor = torch.zeros(1,90,180,768).to(torch.device('cuda'))
    print(f"Time to create tensor: {time.time()-start} rank: {rank}")
    if rank == 0:
        for i in range(mp_size):
            tensor[:,:,:,i*(tensor.shape[3]//mp_size):(i+1)*(tensor.shape[3]//mp_size)] = i
    for dim in range(1,4):
        outHeat = heat(tensor, mp_size, commHEAT, dim)
    start = time.time()
    contig = tensor.permute(3,0,1,2)
    contig = contig.contiguous()
    print(f"Time to create contig tensor: {time.time()-start} rank: {rank} contig shape: {contig.shape}")
    outContig = heat(contig, mp_size, commHEAT, 0)
    outTorch = torchDist(contig, mp_size,0)
    outTorch = outTorch.permute(1,2,3,0)
    outTorch = outTorch.contiguous()
    outContig = outContig.permute(1,2,3,0)
    outContig = outContig.contiguous()
    print(f"Time to create contig tensor: {time.time()-start} rank: {rank} outContig shape: {outContig.shape}, tensor shape: {tensor.shape}")
    close_torch =torch.allclose(tensor, outTorch)
    close_heat=torch.allclose(tensor, outHeat)
    close_both=torch.allclose(outTorch, outHeat)
    close_contig = torch.allclose(tensor, outContig)
    if rank == 0:
        print(f"Rank: {rank} TorchDist: {close_torch} HEAT: {close_heat} Both: {close_both}, Contig: {close_contig}")
        print(f"input: {tensor[0,0:3,0:3,:]}")
        print(f"outTorch: {outTorch[0,0:3,0:3,:]}")
        print(f"outHeat: {outHeat[0,0:3,0:3,:]}")
        print(f"outContig: {outContig[0,0:3,0:3,:]}")
        errors_heat = torch.ne(tensor, outHeat).sum().item()
        errors_torch = torch.ne(tensor, outTorch).sum().item()
        diff = torch.ne(outTorch, outHeat).sum().item()
        contig = torch.ne(tensor, outContig).sum().item()
        print(f"Errors Torch: {errors_torch} Errors Heat: {errors_heat}, Diff Between: {diff}, Contig: {contig}")

    

def scatterTorchDist(input, mp_size,dim):
    rank = dist.get_rank()
    src = rank - rank % mp_size
    split_tensors = list(torch.chunk(input, mp_size, dim=dim))
    received_tensor = torch.empty_like(split_tensors[rank])

    start = time.time()
    dist.scatter(received_tensor,split_tensors if src == rank else None, src=src)
    print(f"TorchDist Scatter time: { time.time()-start} rank: {rank} ")

    return received_tensor

def gatherTorchDist(input, mp_size,dim):
    gathered_tensor = [torch.empty_like(input) for _ in range(mp_size)]

    start = time.time()
    dist.all_gather(gathered_tensor, input)
    print(f"TorchDist Gather time: { time.time()-start} rank: {dist.get_rank()} ")

    gathered_tensor = [t.to(input.device) for t in gathered_tensor]
    grad_output = torch.cat(gathered_tensor, dim=dim) if gathered_tensor is not None else None
    return grad_output, gathered_tensor

def scatterHeat(input, mp_size, commHEAT,dim):
    rank = commHEAT.rank
    split_tensors = split_tensor_along_dim(input, dim, mp_size)
    recvbuf = torch.zeros_like(split_tensors[rank])

    start = time.time()
    commHEAT.Scatterv(input, recvbuf, root=rank-rank%mp_size,axis=dim)
    print(f"HEAT Scatter time: { time.time()-start} rank: {rank} dim: {dim}")

    return recvbuf

def gatherHeat(input,commHEAT,shape,dim):
    recvbuf = torch.empty(shape, dtype=input.dtype, device=input.device)

    start = time.time()
    commHEAT.Allgatherv(input, recvbuf, recv_axis=dim)
    print(f"HEAT Gather time: { time.time()-start} rank: {commHEAT.rank} dim: {dim}")

    return recvbuf

def heat(input, mp_size, commHEAT, dim):
    shape = input.shape
    tensor1 = scatterHeat(input, mp_size, commHEAT, dim)
    tensor2 = gatherHeat(tensor1, commHEAT, shape, dim)
    print(f"heat: input shape: {input.shape}, tensor1 shape: {tensor1.shape}, tensor2 shape: {tensor2.shape}")
    return tensor2

def torchDist(input, mp_size, dim):
    tensor1 = scatterTorchDist(input, mp_size,dim)
    tensor2, gathered_tensors = gatherTorchDist(tensor1, mp_size,dim)
    return tensor2

def split_tensor_along_dim(tensor, dim, num_chunks):

    size = tensor.shape[dim]
        # treat trivial case first
    if num_chunks == 1:
        sections = [size]
    else:
        # first, check if we can split using div-up to balance the load:
        chunk_size = (size + num_chunks - 1) // num_chunks
        last_chunk_size = max(0, size - chunk_size * (num_chunks - 1))
        if last_chunk_size == 0:
            # in this case, the last shard would be empty, split with floor instead:
            chunk_size = size // num_chunks
            last_chunk_size = size - chunk_size * (num_chunks - 1)

        # generate sections list
        sections = [chunk_size for _ in range(num_chunks - 1)] + [last_chunk_size]

    tensor_list = torch.split(tensor, sections, dim=dim)

    return tensor_list

if __name__ == "__main__":
    main(2)
    dist.destroy_process_group()