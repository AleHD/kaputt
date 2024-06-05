import os
import socket
from datetime import timedelta
from typing import Optional

import torch
import datasets
from torch import nn
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed.distributed_c10d import ProcessGroup


# these emulate the payload which will become a M * N * 4-sized tensor below
N = 133_693_441
M = 50
TIMEOUT = timedelta(seconds=30)


class DummyModel(nn.Sequential):
    def __init__(self, params: int = 80):  # in billion.
        self.in_size = int(1_000_000**0.5)
        layers = [nn.Linear(self.in_size, self.in_size)]
        for i in range(params - 1):
            layers += [nn.ReLU(), nn.Linear(self.in_size, self.in_size)]
        layers += [nn.ReLU(), nn.Linear(self.in_size, 256)]
        super().__init__(*layers)


def all_gather_coalesced(  # pylint: disable=function-redefined
    output_tensor_lists: list[list[torch.Tensor]],
    input_tensor_list: list[torch.Tensor],
    group: Optional[ProcessGroup] = None,
    async_op: bool = False,
) -> Optional[torch._C.Future]:
    """
    `torch` has a deprecated version of this method that doesn't work over NCCL.
    All gathers a list of tensors to all processes in a group.

    Args:
        output_tensor_lists (list[list[Tensor]]): Output tensor.
        input_tensor_list (list[Tensor]): List of tensors to all_gather from.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    """
    assert len(output_tensor_lists) > 0
    assert len(input_tensor_list) == len(output_tensor_lists)
    device = input_tensor_list[0].device
    dtype = input_tensor_list[0].dtype
    group_size = len(output_tensor_lists[0])

    assert (
        group_size > 1
    ), "You should probably not call `all_gather_coalesced` with a single rank, as it copies data over"

    for input_tensor in input_tensor_list:
        assert device == input_tensor.device
        assert dtype == input_tensor.dtype

    for output_tensor_list in output_tensor_lists:
        assert len(output_tensor_list) == group_size
        for output_tensor in output_tensor_list:
            assert device == output_tensor.device
            assert dtype == output_tensor.dtype

    # Invert from `[param_idx][group_rank]` to `[group_rank][param_idx]`
    output_tensor_lists = [
        [output_tensor_list[group_rank] for output_tensor_list in output_tensor_lists]
        for group_rank in range(group_size)
    ]

    input_tensor_buffer = torch._utils._flatten_dense_tensors(input_tensor_list)
    output_tensor_buffer_list = [
        torch._utils._flatten_dense_tensors(output_tensor_list) for output_tensor_list in output_tensor_lists
    ]

    work = torch.distributed.all_gather(output_tensor_buffer_list, input_tensor_buffer, group=group, async_op=async_op)

    def update_output():
        for original_buffer_list, gathered_buffer_tensor in zip(output_tensor_lists, output_tensor_buffer_list):
            for original_buffer, gathered_buffer in zip(
                original_buffer_list,
                torch._utils._unflatten_dense_tensors(gathered_buffer_tensor, original_buffer_list),
            ):
                original_buffer.copy_(gathered_buffer)

    if async_op is True:
        return work.get_future().then(lambda fut: update_output())
    else:
        # No need to run `work.wait()` since `dist.reduce_scatter` already waits
        update_output()



def main():
    # Init distributed.
    local_rank = int(os.environ["LOCAL_RANK"])
    print(f"Starting in {socket.gethostname()} with rank {local_rank}")
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.float32
    torch.distributed.init_process_group("nccl", timeout=TIMEOUT)
    torch.cuda.set_device(local_rank)


    for i in range(10):
        # All reduce.
        tensor = torch.randn(N, M, dtype=dtype, device=device)
        torch.distributed.all_reduce(tensor)
        torch.cuda.synchronize()

        # All gather.
        n_lists = 10
        K = N//(torch.distributed.get_world_size()*n_lists*10)
        inputs = [torch.randn(K, M, dtype=dtype, device=device) for _ in range(n_lists)]
        outputs = [[torch.empty(K, M, dtype=dtype, device=device)
                    for _ in range(torch.distributed.get_world_size())]
                   for _ in range(n_lists)]
        all_gather_coalesced(output_tensor_lists=outputs, 
                             input_tensor_list=inputs)
        torch.cuda.synchronize()

    # Dummy DDP.
    model = DummyModel().to(device)
    in_size = model.in_size
    model = nn.parallel.DistributedDataParallel(model)
    optimizer = ZeroRedundancyOptimizer(
        model.parameters(),
        optimizer_class=torch.optim.SGD,
        lr=0.000
    )
    criterion = nn.MSELoss()
    for i in range(10):
        x = torch.randn(1, in_size, dtype=dtype, device=device)
        y = model(x)
        loss = criterion(y, torch.randn_like(y))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()

    # Read dataset.
    dataset = datasets.load_dataset("DKYoon/SlimPajama-6B", split="train")
    for i, doc in zip(range(10), dataset):
        pass

    torch.cuda.synchronize()
    torch.distributed.barrier()
    print("Success in", socket.gethostname())


if __name__ == "__main__":
    main()
