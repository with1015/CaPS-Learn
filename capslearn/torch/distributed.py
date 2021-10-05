import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc

import os


# Distributeddataparallel test

class DistributedDataParallel(nn.Module):

    def __init__(self, model=None,
                        rank=0,
                        world_size=0,
                        num_gpus=0,
                        device_ids=None,
                        broadcast_buffers=True):

        super().__init__()
        self.model = model
        self.root  = 0
        self.world_size = world_size
        self.rank  = rank
        self.device_ids = device_ids
        self.require_forward_param_sync = broadcast_buffers
        self._set_up()


    def _broadcast_parameters(params, root_rank):
        if self.rank == 0:
            # TODO: Push parameters
            for param in params:
                dist.broadcast(tensor=param, src=self.root, group=None, async_op=False)
        else:
            # TODO: Pull parameters
            pass

    def _sync_params(self):
        with torch.no_grad():
            # TODO : Braodcast parameters with synchronization
            self._broadcast_parameters()
            pass


    def _set_up():
        os.environ['MASTER_ADDR'] = None
        os.environ['MASTER_PORT'] = '50051'
        dist.init_process_group(backend='gloo', rank=self.rank, world_size=self.world_size)


    def _clean_up():
        dist.destroy_process_group()


    def forward(self, x):
        if self.require_forward_param_sync:
            self._sync_params()
        return self.model(x)



# Parameter server test

def call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


def remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    print(rref.owner())
    return rpc.rpc_sync(rref.owner(), call_method, args=args, kwargs=kwargs)


def run_parameter_server(rank, world_size):
    rpc.init_rpc(name="parameter_server", rank=rank, world_size=world_size)
    print("Init RPC parameter server done.")
    rpc.shutdown()


def run_worker(rank, world_size, num_gpus, train_loader, test_loader, training_fn):
    rpc.init_rpc(name=f"trainer_{rank}", rank=rank, world_size=world_size)
    training_fn(rank, num_gpus, train_loader, test_loader)
    rpc.shutdown()



class ParameterServer(nn.Module):

    def __init__(self, model=None, num_gpus=0):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and num_gpus > 0 else "cpu")
        self.model = model.to(self.device)


    def forward(self, x):
        x = x.to(self.device)
        result = self.model(x)
        result = result.to("cpu")
        return result


    def get_dist_gradients(self, context_id):
        grads = dist_autograd.get_gradients(context_id)
        cpu_grads = {}

        for k, v in grads.items():
            k_cpu, v_cpu = k.to("cpu"), v.to("cpu")
            cpu_grads[k_cpu] = v_cpu

        return cpu_grads


    def get_param_rrefs(self):
        param_rrefs = [rpc.RRef(param) for param in self.model.parameters()]
        return param_rrefs

