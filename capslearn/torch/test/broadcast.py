import os
import torch
import torch.distributed as dist

from torch.multiprocessing import Process

def run(rank, size):
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
    dist.broadcast(tensor, src=0)
    print('Rank', rank, ' has data ', tensor.item())


def init_processes(rank, size, fn, backend='gloo'):
    os.environ['MASTER_ADDR'] = 'ib041'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)



if __name__ == "__main__":
    processes = []
    p = Process(target=init_processes, args=(0, 2, run))
    p.start()
    processes.append(p)

    for p in processes:
        p.join()
