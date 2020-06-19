import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

def example():#rank, world_size):
    if True:
        torch.cuda.set_device(0)
        dist.init_process_group(backend='gloo',
                                             init_method='env://')
        print(torch.distributed.get_world_size())
        print(torch.cuda.device_count())

    # create default process group
    # dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # create local model
    model = nn.Linear(1000, 10).to(0)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    for j in range(100):
        # forward pass
        outputs = ddp_model(torch.randn(20, 1000).to(0))
        labels = torch.randn(20, 10).to(0)
        # backward pass
        loss_fn(outputs, labels).backward()
        # update parameters
        optimizer.step()
        optimizer.zero_grad()

def main():
    example()
    # world_size = 3
    # mp.spawn(example,
    #     args=(world_size,),
    #     nprocs=world_size,
    #     join=True)

if __name__=="__main__":
    duration = []
    for i in range(10):
        start = time.time()
        main()
        end = time.time()
        duration.append(end-start)
    print(f'Runtimes: {duration}')
    print(f'Average run took {sum(duration) / len(duration)}')
