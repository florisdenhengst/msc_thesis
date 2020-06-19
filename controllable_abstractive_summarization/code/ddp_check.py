import argparse

import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

parser.add_argument('--distributed', action='store_true', help='enables distributed processes')
parser.add_argument('--local_rank', default=0, type=int, help='number of distributed processes')
parser.add_argument('--dist_backend', default='gloo', type=str, help='distributed backend')


def example(rank):#rank, world_size):
    # create local model
    model = nn.Linear(1000, 10).to(rank)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    for j in range(100):
        # forward pass
        outputs = ddp_model(torch.randn(20, 1000).to(rank))
        labels = torch.randn(20, 10).to(rank)
        # backward pass
        loss_fn(outputs, labels).backward()
        # update parameters
        optimizer.step()
        optimizer.zero_grad()

def main():
    example(dist.get_rank())
    # world_size = 3
    # mp.spawn(example,
    #     args=(world_size,),
    #     nprocs=world_size,
    #     join=True)

if __name__=="__main__":
    args = parser.parse_args()
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method='env://')
        print("Initialized Rank:", dist.get_rank())
        print(torch.distributed.get_world_size())
        print(torch.cuda.device_count())


    duration = []
    for i in range(10):
        start = time.time()
        main()
        end = time.time()
        duration.append(end-start)
    print(f'Runtimes: {duration}')
    print(f'Average run took {sum(duration) / len(duration)}')



