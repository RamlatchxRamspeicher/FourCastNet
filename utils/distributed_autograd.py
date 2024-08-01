# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import torch
import torch.distributed as dist
from typing import List, Optional
from utils.comm import get_world_size, get_world_rank

def compute_split_shapes(size: int, num_chunks: int) -> List[int]:
    # treat trivial case first
    if num_chunks == 1:
        return [size]

    # first, check if we can split using div-up to balance the load:
    chunk_size = (size + num_chunks - 1) // num_chunks
    last_chunk_size = max(0, size - chunk_size * (num_chunks - 1))
    if last_chunk_size == 0:
        # in this case, the last shard would be empty, split with floor instead:
        chunk_size = size // num_chunks
        last_chunk_size = size - chunk_size * (num_chunks - 1)

    # generate sections list
    sections = [chunk_size for _ in range(num_chunks - 1)] + [last_chunk_size]

    return sections

def split_tensor_along_dim(tensor, dim, num_chunks):
    if dim >= tensor.dim():
        raise ValueError(
            f"Error, tensor dimension is {tensor.dim()} which cannot be split along {dim}"
        )
    if tensor.shape[dim] < num_chunks:
        raise ValueError(
            "Error, cannot split dim {dim} of size {tensor.shape[dim]} into \
        {num_chunks} chunks. Empty slices are currently not supported."
        )

    # get split
    sections = compute_split_shapes(tensor.shape[dim], num_chunks)
    tensor_list = torch.split(tensor, sections, dim=dim)

    return tensor_list

def get_memory_format(tensor):
    """Gets format for tensor"""
    if tensor.is_contiguous(memory_format=torch.channels_last):
        return torch.channels_last
    else:
        return torch.contiguous_format

def _reduce(input_, use_fp32=True, group=None):  # pragma: no cover
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if get_world_size() == 1:
        return input_

    # All-reduce, use_fp32 only relevant for lower precisions
    # if input is already in double precision, nothing changes
    if use_fp32 and (input_.dtype.itemsize < 4) and input_.dtype.is_floating_point:
        dtype = input_.dtype
        inputf_ = input_.float()
        dist.all_reduce(inputf_, group=group)
        input_ = inputf_.to(dtype)
    else:
        dist.all_reduce(input_, group=group)

    return input_

def _split(input_, dim_, group=None):  # pragma: no cover
    """Split the tensor along its last dimension and keep the corresponding slice."""
    # get input format
    input_format = get_memory_format(input_)

    # Bypass the function if we are using only 1 GPU.
    comm_size = get_world_size()
    if comm_size == 1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_dim(input_, dim_, comm_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = get_world_rank()
    output = input_list[rank].contiguous(memory_format=input_format)

    return output

def all_gather_v_wrapper(
    tensor: torch.Tensor,
    sizes: Optional[List[int]] = None,
    dim: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:  # pragma: no cover
    """
    Implements a distributed AllGatherV primitive. It is based
    on the idea of a single global tensor which is distributed along
    a specified dimension into chunks of variable size.
    This primitive gathers all local tensors from each rank into the
    full global tensor onto each rank.

    Parameters
    ----------
    tensor : "torch.Tensor"
        local tensor on each rank
    sizes : List[int], optional
        list of the sizes of each chunk on each rank along distributed dimension,
        valid and set on each rank, by default None
    dim : int, optional
        dimension along which global tensor is distributed, by default 0
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    torch.Tensor
        full global tensor, valid on each rank
    """

    comm_size = get_world_size()

    if (sizes is not None) and (len(sizes) != comm_size):
        raise ValueError()
    if dim >= tensor.dim():
        raise ValueError()

    if comm_size == 1:
        return tensor

    tensor_shape = list(tensor.shape)
    tensor_format = get_memory_format(tensor)

    if sizes is not None:
        tensor_list = [None] * comm_size

        for src in range(comm_size):
            tensor_shape[dim] = sizes[src]
            tensor_list[src] = torch.empty(
                tensor_shape,
                dtype=tensor.dtype,
                device=tensor.device,
            )
    else:
        # assume equal shape on all ranks
        tensor_list = [torch.empty_like(tensor) for _ in range(comm_size)]

    dist.all_gather(tensor_list, tensor, group=group)

    output = torch.cat(tensor_list, dim=dim).contiguous(memory_format=tensor_format)

    return output

class _CopyToParallelRegion(torch.autograd.Function):
    """Pass the input to the parallel region"""

    @staticmethod
    def symbolic(graph, input_, group_):  # pragma: no cover
        return input_

    @staticmethod
    def forward(ctx, input_, group_):  # pragma: no cover
        ctx.group = group_
        return input_

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        return _reduce(grad_output, group=ctx.group), None


class _ReduceFromParallelRegion(torch.autograd.Function):
    """All-reduce the input from the parallel region"""

    @staticmethod
    def symbolic(graph, input_, group_):  # pragma: no cover
        return _reduce(input_, group=None)

    @staticmethod
    def forward(ctx, input_, group_):  # pragma: no cover
        return _reduce(input_, group=None)

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        return grad_output, None


class _ScatterToParallelRegion(torch.autograd.Function):
    """Split the input and keep only the chunk corresponding to the rank."""

    @staticmethod
    def symbolic(graph, input_, dim_, group_):  # pragma: no cover
        return _split(input_, dim_, group=None)

    @staticmethod
    def forward(ctx, input_, dim_, group_):  # pragma: no cover
        ctx.dim = dim_
        ctx.group = group_
        ctx.split_shapes = compute_split_shapes(
            input_.shape[dim_], get_world_size()
        )
        return _split(input_, dim_, group=None)

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        return (
            all_gather_v_wrapper(
                grad_output,
                ctx.split_shapes,
                ctx.dim,
                group=ctx.group,
            ),
            None,
            None,
        )


class _GatherFromParallelRegion(torch.autograd.Function):
    """Gather the input from parallel region and concatenate."""

    @staticmethod
    def symbolic(graph, input_, dim_, group_, shapes_):  # pragma: no cover
        return all_gather_v_wrapper(
            input_, shapes_, dim_, group=None
        )

    @staticmethod
    def forward(ctx, input_, dim_, shapes_, group_):  # pragma: no cover
        ctx.dim = dim_
        ctx.group = group_
        return all_gather_v_wrapper(
            input_, shapes_, dim_, group=group_
        )

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        return (
            _split(grad_output, ctx.dim, group=ctx.group),
            None,
            None,
            None,
        )


# -----------------
# Helper functions.
# -----------------
def copy_to_parallel_region(input, group):  # pragma: no cover
    """Copy input"""
    return _CopyToParallelRegion.apply(input, group)


def reduce_from_parallel_region(input, group):  # pragma: no cover
    """All-reduce the input from the matmul parallel region."""
    return _ReduceFromParallelRegion.apply(input, group)


def scatter_to_parallel_region(input, dim, group):  # pragma: no cover
    """Split the input and keep only the corresponding chuck to the rank."""
    return _ScatterToParallelRegion.apply(input, dim, group)


def gather_from_parallel_region(input, dim, shapes, group):  # pragma: no cover
    """Gather the input from matmul parallel region and concatenate."""
    return _GatherFromParallelRegion.apply(input, dim, shapes, group)
